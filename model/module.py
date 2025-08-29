import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_hw_only(x):
    # x: [B, T, C, H, W]  -> downsample H,W by 2 using 2D pooling
    B, Tn, Cn, H, W = x.shape
    x = x.reshape(B, Tn*Cn, H, W)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)  # no dtype change here
    x = F.avg_pool2d(x, kernel_size=2, stride=2)  # no dtype change here
    x = F.avg_pool2d(x, kernel_size=2, stride=2)  # no dtype change here
    H2, W2 = x.shape[-2:]
    x = x.reshape(B, Tn, Cn, H2, W2)
    return x

def pad_to_multiple(x, multiple=16):
    # x: (B, T, C, H, W) or (B, C, H, W)
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    # pad bottom and right (keep top-left aligned)
    padding = (0, pad_w, 0, pad_h)  
    return F.pad(x, padding, mode="constant", value=0)

class EncoderConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(EncoderConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class TranslatorConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, groups):
        super(TranslatorConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size//2,
                              groups=groups)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels,
                 groups,
                 incep_kernel_sizes=[3, 5, 7, 11]):
        super(Inception, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels//2,
                               kernel_size=1,
                               stride=1, padding=0)

        incep_layers = []
        for kernel_size in incep_kernel_sizes:
            conv = TranslatorConv(
                in_channels=out_channels//2,
                out_channels=out_channels,
                kernel_size=kernel_size,
                groups=groups
            )
            incep_layers.append(conv)
        self.incep_layers = nn.Sequential(*incep_layers)

    def forward(self, x):
        x = self.conv1(x)

        y = 0
        for layer in self.incep_layers:
            y += layer(x)

        return y


class DecoderConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DecoderConv2d, self).__init__()

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=3, padding=1,
                                         stride=stride, output_padding=stride//2)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ConvGRUCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3, padding=1):
        super().__init__()
        self.in_ch = in_ch
        self.hid_ch = hid_ch
        self.gates = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, k, padding=padding)
        self.cand  = nn.Conv2d(in_ch + hid_ch, hid_ch, k, padding=padding)

    def forward(self, x, h=None):
        # x: [B, C_in, H, W], h: [B, C_hid, H, W] or None
        if h is None:
            h = torch.zeros(x.size(0), self.hid_ch, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        z_r = self.gates(torch.cat([x, h], dim=1))
        z, r = torch.chunk(z_r, 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)
        hcand = torch.tanh(self.cand(torch.cat([x, r * h], dim=1)))
        h_new = (1 - z) * h + z * hcand
        return h_new

def sinusoidal_embedding(t_idx: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    t_idx: [B] or scalar tensor with integer time indices
    d_model: embedding dimension (must be even for sin/cos pairs)
    returns: [B, d_model] sinusoidal embedding
    """
    if t_idx.dim() == 0:
        t_idx = t_idx.unsqueeze(0)  # make it [1]

    device = t_idx.device
    half_dim = d_model // 2
    # log space frequencies
    freqs = torch.exp(
        -torch.arange(0, half_dim, dtype=torch.float32, device=device)
        * (torch.log(torch.tensor(10000.0)) / half_dim)
    )
    # outer product → [B, half_dim]
    angles = t_idx.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [B, d_model]

    if d_model % 2 == 1:  # pad if odd
        emb = torch.nn.functional.pad(emb, (0,1))
    return emb

class TimeEmb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d, d*4), nn.SiLU(), nn.Linear(d*4, d))
    def forward(self, t_idx, d_model):
        # sinusoidal pe -> [B, d], then MLP
        pe = sinusoidal_embedding(t_idx, d_model)  # implement or borrow
        return self.proj(pe)


class Frame2Enc(nn.Module):
    def __init__(self, C_img, C_enc):
        super().__init__()
        # Example: two strided convs to go from (H,W) -> (Hs,Ws)
        self.net = nn.Sequential(
            nn.Conv2d(C_img, C_enc//2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(C_enc//2, C_enc, 3, stride=2, padding=1),
        )
    def forward(self, y):  # y: [B,T,C,H,W]
        B, T, C, H, W = y.shape
        y = y.view(B*T, C, H, W)
        z = self.net(y)  # [B*T, C_enc, Hs, Ws]
        # If scale mismatch, adjust strides or add pooling.
        return z.view(B, T, -1, z.shape[-2], z.shape[-1])  # [B,T,C_enc,Hs,Ws]


