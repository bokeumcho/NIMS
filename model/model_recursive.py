import torch
import torch.nn as nn

from .module import EncoderConv, Inception, DecoderConv2d, ConvGRUCell, TimeEmb, Frame2Enc
from .model import *

class TranslatorBase(nn.Module):
    def __init__(self, in_channels, hid_channels, Nt,
                 groups, incep_kernel_sizes):
        super(TranslatorBase, self).__init__()
        self.Nt = Nt

        enc_layers = [Inception(in_channels, hid_channels, groups=groups,
                                incep_kernel_sizes=incep_kernel_sizes)]
        for _ in range(Nt - 1):
            enc_layers.append(
                Inception(hid_channels, hid_channels, groups=groups,
                          incep_kernel_sizes=incep_kernel_sizes)
            )

        dec_layers = [Inception(in_channels=hid_channels, out_channels=hid_channels,
                                groups=groups, incep_kernel_sizes=incep_kernel_sizes)]
        for _ in range(Nt - 2):
            dec_layers.append(
                Inception(2*hid_channels, hid_channels, groups=groups,
                          incep_kernel_sizes=incep_kernel_sizes)
            )
        dec_layers.append(Inception(2*hid_channels, in_channels, groups=groups,
                                    incep_kernel_sizes=incep_kernel_sizes))

        self.enc_layers = nn.Sequential(*enc_layers)
        self.dec_layers = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C_, H_, W_ = x.shape
        x = x.reshape(B, T * C_, H_, W_)

        skips = []
        for i in range(self.Nt):
            x = self.enc_layers[i](x)
            if i < self.Nt - 1:
                skips.append(x)

        x = self.dec_layers[0](x)
        for i in range(1, self.Nt):
            x = torch.cat([x, skips[-i]], dim=1)
            x = self.dec_layers[i](x)

        x = x.contiguous().view(B, T, C_, H_, W_) # only changed func. all logic same
        return x

class DecoderBase(nn.Module):
    def __init__(self, C, C_enc, Ns):
        super().__init__()
        self.C = C
        self.Ns = Ns
        strides = stride_generator(Ns)[::-1]

        # Name layers explicitly to avoid off-by-one confusions
        # self.stem = nn.ModuleList([DecoderConv2d(C_enc, C_enc, stride=s) for s in strides[:-1]]) # [:-1]
        # self.fuse = DecoderConv2d(3 * C_enc, C_enc, stride=strides[-1])  # x + enc1 + recent
        
        # 1) Fuse first: no spatial change
        self.fuse = DecoderConv2d(3 * C_enc, C_enc, stride=1)

        # 2) Then upsample through all scales
        self.stem = nn.ModuleList([DecoderConv2d(C_enc, C_enc, stride=s) for s in strides])

        self.readout = nn.Conv2d(C_enc, C, kernel_size=1, stride=1, padding=0)

    def forward(self, x_ctx_seq, enc1_seq, recent_seq, H, W):
        # all: [B, T, C_enc, Hs, Ws]
        B, T, C_, Hs, Ws = x_ctx_seq.shape

        x = x_ctx_seq.reshape(B*T, C_, Hs, Ws)
        e  = enc1_seq.reshape(B*T, C_, Hs, Ws)
        r = recent_seq.reshape(B*T, C_, Hs, Ws)
        print(x.shape, e.shape, r.shape)

        x = torch.cat([x, e, r], dim=1)     # [B*T, 3*C_enc, Hs, Ws]
        x = self.fuse(x)                    # [B*T, C_enc, Hs, Ws]

        # up blocks except last (which we use as fuse)
        for block in self.stem:
            x = block(x) # [B*T, C_enc, Hs, Ws]
        print(x.shape)
        

        x = self.readout(x)                 # [B*T, C, H, W]
        x = x.reshape(B, T, self.C, H, W)
        
        return x

class DecoderOnlyAR(nn.Module):
    def __init__(self, base_decoder, T, C_img, C_enc, hid_ch=None, t_dim=32):
        super().__init__()
        self.T = T

        # context GRU: consumes Z_ctx_flat [B, T*C_enc, Hs, Ws]
        self.gru_ctx = ConvGRUCell(T * C_enc, hid_ch or C_enc, k=3, padding=1)
        self.to_dec = nn.Conv2d(hid_ch or C_enc, C_enc, 1)  # to per-time latent

        self.time_emb = TimeEmb(t_dim)
        self.to_film = nn.Linear(t_dim, 2 * C_enc)

        self.base = base_decoder  # expects per-time latents
        self.frame2enc_sameC = Frame2Enc(C_img=C_enc, C_enc=C_enc)
        self.frame2enc = Frame2Enc(C_img=C_img, C_enc=C_enc)

    @torch.no_grad()
    def _ss_mix(self, pred, gt, p):
        if gt is None or p <= 0.0: return pred
        gt = gt.to(device=pred.device, dtype=pred.dtype)    # 🔧 belt-and-suspenders

        if p >= 1.0: return gt
        B, T, C, H, W = pred.shape
        mask = (torch.rand(B, T, 1, 1, 1, device=pred.device) < p).float()
        return mask * gt + (1.0 - mask) * pred

    def rollout(self, Z_ctx_seq, enc1_seq, horizon, detach_between_steps=True, tf_mode=False, teacher=None, ss_ratio=0.0):
        """
        Z_ctx_seq:  [B, T, C_enc, Hs, Ws]
        enc1_seq:   [B*T, C_enc, H, W]
        teacher:    [B, horizon, C_img, H, W]  (full future)
        returns:    [B, horizon, C_img, H, W]
        """
        B, T, C_enc, Hs, Ws = Z_ctx_seq.shape
        _, _, H, W = enc1_seq.shape
        print('\n rollout', Z_ctx_seq.shape, enc1_seq.shape)

        device = Z_ctx_seq.device
        dtype  = Z_ctx_seq.dtype

        enc1_seq = enc1_seq.to(device=device, dtype=dtype)

        # init recent encodings as zeros
        recent_seq = torch.zeros(B, self.T, C_enc, Hs, Ws, device=device, dtype=dtype)
        enc1_seq_mapped = self.frame2enc_sameC(enc1_seq.reshape(B, T, C_enc, H, W))

        h_ctx = None
        outs = []
        num_blocks = (horizon + self.T - 1) // self.T

        for k in range(num_blocks):  # block index 0..num_blocks-1
            # 1) evolve context state
            # h_ctx = self.gru_ctx(Z_ctx_flat, h_ctx)      # [B, C_hid, Hs, Ws]
            # z_base = self.to_dec(h_ctx)                  # [B, C_enc, Hs, Ws]

            # time conditioning (block index)
            te = self.time_emb(
                torch.full((B,), k, device=Z_ctx_seq.device, dtype=torch.long),
                d_model=self.to_film.in_features
            )                                # [B, t_dim]

            gamma, beta = self.to_film(te).chunk(2, dim=-1)  # [B, C_enc] each

            # IMPORTANT: add the time dim (1) so it broadcasts over T,H,W
            gamma = gamma.reshape(B, 1, C_enc, 1, 1)
            beta  = beta.reshape( B, 1, C_enc, 1, 1)

            z_k_seq = Z_ctx_seq * (1 + gamma) + beta        # [B, T, C_enc, Hs, Ws]
            
            # 2) make a per-time latent sequence for this block
            # z_base_seq = z_base.unsqueeze(1).expand(B, self.T, C_enc, Hs, Ws)

            # 3) decode T frames using fixed context + skip + recent
            print('bf concat', z_k_seq.shape, enc1_seq_mapped.shape, recent_seq.shape)

            y_block = self.base(z_k_seq, enc1_seq_mapped, recent_seq, H, W)   # [B, T, C_img, Hs, Ws]
            outs.append(y_block)

            # 4) scheduled sampling mix for building next 'recent_seq'
            if tf_mode and (teacher is not None):
                t0, t1 = k * self.T, min((k+1) * self.T, horizon)  # slice of ground truth
                teacher_block = teacher[:, t0:t1].to(device=device, dtype=self.frame2enc.weight.dtype
                                if hasattr(self.frame2enc, "weight") else dtype)

                # pad to T if last block is short
                if t1 - t0 < self.T:
                    pad = self.T - (t1 - t0)
                    teacher_block = torch.cat([teacher_block, y_block[:, (t1 - t0):t1 - t0 + pad]], dim=1)
                mixed = self._ss_mix(y_block, teacher_block, ss_ratio)  # [B,T,C,H,W]
            else:
                mixed = y_block

            # 5) update recent encoding for next block
            recent_seq = self.frame2enc(mixed)  # [B, T, C_enc, Hs, Ws]

            if detach_between_steps:
                recent_seq = recent_seq.detach()

        y = torch.cat(outs, dim=1)[:, :horizon]  # [B, horizon, C_img, H, W]
        return y

class SimVP_AR_Decoder(nn.Module):
    def __init__(self, T, C_img, C_enc, C_hid, Ns, Nt, groups, horizon, incep_kernel_sizes=[3, 5, 7, 11]):
        super().__init__()
        self.T = T
        self.horizon = horizon
        self.encoder = Encoder(C_img, C_enc, Ns)       # must return per-time enc + skip per-time
        self.translator = Translator(T*C_enc, C_hid, Nt, groups, incep_kernel_sizes)
        self.decoder_base = DecoderBase(C=C_img, C_enc=C_enc, Ns=Ns)
        self.decoder = DecoderOnlyAR(self.decoder_base, T, C_img, C_enc, hid_ch=C_enc)

    def forward(self, x, detach_between_steps=True, tf_mode=False, teacher=None, ss_ratio=0.0):
        # x: [B, T, C_img, H, W]
        Z_seq, enc1_seq = self.encoder(x)               # recommend: both as [B, T, C_enc, Hs, Ws]
        # If your encoder returns [B, T*C_enc, Hs, Ws], reshape to per-time.
        if Z_seq.dim() == 4:
            B, TC, Hs, Ws = Z_seq.shape
            C_enc = TC // self.T
            Z_seq = Z_seq.reshape(B, self.T, C_enc, Hs, Ws)

        # Translator expects [B, T, C_enc, Hs, Ws]
        Z_ctx_flat = self.translator(Z_seq)             # [B, T*C_enc, Hs, Ws]

        y = self.decoder.rollout(Z_ctx_flat, enc1_seq, self.horizon,
                                 detach_between_steps=detach_between_steps,
                                 tf_mode=tf_mode, teacher=teacher)
        return y                                         # [B, horizon, C_img, H, W]
