import torch
import torch.nn as nn

from .module import EncoderConv, Inception, DecoderConv2d
from .model import Encoder, stride_generator

class TranslatorK(nn.Module):  # >>> NEW
    def __init__(self, T, k, C_enc, C_hid, Nt, groups, incep_kernel_sizes):
        super().__init__()
        self.T = T
        self.k = k
        self.C_enc = C_enc
        self.Nt = Nt

        in_channels  = T * C_enc              # encode past T
        out_channels = (k * T) * C_enc        # >>> CHANGED: decode k·T

        enc_layers = [Inception(in_channels, C_hid, groups=groups,
                                incep_kernel_sizes=incep_kernel_sizes)]
        for _ in range(Nt - 1):
            enc_layers.append(Inception(C_hid, C_hid, groups=groups,
                                        incep_kernel_sizes=incep_kernel_sizes))

        dec_layers = [Inception(in_channels=C_hid, out_channels=C_hid,
                                groups=groups, incep_kernel_sizes=incep_kernel_sizes)]
        for _ in range(Nt - 2):
            dec_layers.append(Inception(2*C_hid, C_hid, groups=groups,
                                        incep_kernel_sizes=incep_kernel_sizes))
        dec_layers.append(Inception(2*C_hid, out_channels, groups=groups,
                                    incep_kernel_sizes=incep_kernel_sizes))  # >>> CHANGED

        self.enc_layers = nn.Sequential(*enc_layers)
        self.dec_layers = nn.Sequential(*dec_layers)

    def forward(self, z_seq):  # z_seq: [B, T, C_enc, Hs, Ws]
        B, T, C, Hs, Ws = z_seq.shape
        assert T == self.T and C == self.C_enc
        x = z_seq.reshape(B, T*C, Hs, Ws)

        skips = []
        for i in range(self.Nt):
            x = self.enc_layers[i](x)
            if i < self.Nt - 1:
                skips.append(x)

        x = self.dec_layers[0](x)
        for i in range(1, self.Nt):
            x = torch.cat([x, skips[-i]], dim=1)
            x = self.dec_layers[i](x)

        # reshape to per-time (k·T) latents
        x = x.reshape(B, self.k*self.T, self.C_enc, Hs, Ws)  # >>> CHANGED
        return x  # [B, kT, C_enc, Hs, Ws]

class DecoderK(nn.Module):  # >>> NEW
    def __init__(self, C_out, C_enc, Ns):
        super().__init__()
        self.C_out = C_out
        self.C_enc = C_enc
        strides = stride_generator(Ns)[::-1]

        # split stem vs fuse to avoid off-by-one issues
        self.stem = nn.ModuleList([DecoderConv2d(C_enc, C_enc, stride=s) for s in strides[:-1]])
        self.fuse = DecoderConv2d(2*C_enc, C_enc, stride=strides[-1])  # x + skip
        self.readout = nn.Conv2d(C_enc, C_out, kernel_size=1, stride=1, padding=0)

    def forward(self, z_seq, enc1_seq=None):
        """
        z_seq:    [B, kT, C_enc, Hs, Ws]
        enc1_seq: [B, kT, C_enc, Hs, Ws] or None (already broadcast)
        """
        B, S, C, Hs, Ws = z_seq.shape   # S = kT
        x = z_seq.reshape(B*S, C, Hs, Ws)

        for blk in self.stem:
            x = blk(x)

        if enc1_seq is not None:
            e = enc1_seq.reshape(B*S, C, x.shape[-2], x.shape[-1])
            x = torch.cat([x, e], dim=1)
            x = self.fuse(x)
        else:
            # If you ever skip skip-conn, replace fuse with a C_enc->C_enc block here.
            x = self.fuse(torch.cat([x, torch.zeros_like(x)], dim=1))

        y = self.readout(x)                    # [B*S, C_out, H, W]
        y = y.reshape(B, S, self.C_out, y.shape[-2], y.shape[-1])  # [B,kT,C_out,H,W]
        return y

class SimVP_kT(nn.Module):  # >>> NEW
    def __init__(self, T, k, C_in, C_out, C_enc, C_hid, Ns, Nt,
                 groups, incep_kernel_sizes=[3,5,7,11]):
        super().__init__()
        self.T = T
        self.k = k
        self.encoder = Encoder(C_in, C_enc, Ns)
        self.translator = TranslatorK(T, k, C_enc, C_hid, Nt, groups, incep_kernel_sizes)
        self.decoder   = DecoderK(C_out, C_enc, Ns)

    def forward(self, x):  # x: [B,T,C_in,H,W]
        B,T,C,H,W = x.shape
        assert T == self.T

        z_seq, enc1_flat = self.encoder(x)                  # z_seq: [B,T,C_enc,Hs,Ws], enc1_flat: [B*T,C_enc,H1,W1]
        # reshape enc1 to [B,T,C_enc,H1,W1] then pool over T and broadcast to kT
        H1, W1 = enc1_flat.shape[-2:]
        enc1_seq = enc1_flat.reshape(B, T, -1, H1, W1)         # >>> CHANGED
        # enc1_mean = enc1_seq.mean(dim=1, keepdim=True)      # [B,1,C_enc,H1,W1]
        enc1_bcast = enc1_seq.repeat(1, self.k, 1, 1, 1)      # [B, k*T, C_enc, H1, W1]

        z_k_seq = self.translator(z_seq)                    # [B,kT,C_enc,Hs,Ws]
        y = self.decoder(z_k_seq, enc1_bcast)               # [B,kT,C_out,H,W]
        return y
