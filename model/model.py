import torch
import torch.nn as nn

from .module import EncoderConv, Inception, DecoderConv2d


def stride_generator(N):
    strides = [1, 2]*100
    return strides[:N]


class Encoder(nn.Module):
    def __init__(self, C, C_enc, Ns):
        super(Encoder, self).__init__()
        self.C_enc = C_enc
        strides = stride_generator(Ns)

        self.layers = nn.Sequential(
            EncoderConv(C, C_enc, stride=strides[0]),
            *[EncoderConv(C_enc, C_enc, stride=strides[layer_idx])
              for layer_idx in range(1, Ns)]
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        enc1_x = self.layers[0](x)

        x = enc1_x
        
        for layer in self.layers[1:]:
            x = layer(x)

        _, _, H_, W_ = x.shape
        x = x.reshape(B, T, self.C_enc, H_, W_)
        return x, enc1_x


class Translator(nn.Module):
    def __init__(self, in_channels, hid_channels, Nt,
                 groups, incep_kernel_sizes):
        super(Translator, self).__init__()
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

        x = x.reshape(B, T, C_, H_, W_)
        return x


class Decoder(nn.Module):
    def __init__(self, C, C_enc, Ns):
        super(Decoder, self).__init__()
        self.C = C
        self.Ns = Ns
        strides = stride_generator(Ns)[::-1]

        self.layers = [
            *[DecoderConv2d(C_enc, C_enc, stride=s)
              for s in strides[:-1]],
            DecoderConv2d(2*C_enc, C_enc, stride=strides[-1])
        ]
        self.layers = nn.Sequential(*self.layers)

        self.readout = nn.Conv2d(C_enc, C, kernel_size=1,
                                 stride=1, padding=0)

    def forward(self, x, enc1_x):
        B, T, C_, H_, W_ = x.shape
        x = x.reshape(B*T, C_, H_, W_)

        for i in range(self.Ns - 1):
            x = self.layers[i](x)

        x = torch.cat([x, enc1_x], dim=1)
        x = self.layers[-1](x)

        x = self.readout(x)
        return x


class SimVP(nn.Module):
    def __init__(self, T, C, C_enc, C_hid, Ns, Nt,
                 groups, incep_kernel_sizes=[3, 5, 7, 11]):
        super(SimVP, self).__init__()

        self.encoder = Encoder(C, C_enc, Ns)
        self.translator = Translator(T*C_enc, C_hid, Nt, groups, incep_kernel_sizes)
        self.decoder = Decoder(C, C_enc, Ns)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x, enc1_x = self.encoder(x)
        x = self.translator(x)
        x = self.decoder(x, enc1_x)

        x = x.reshape(B, T, C, H, W)
        return x

# # add Satelite-to-Rainfall model


# class realModel()
#     video_pred = simvp()
#     final_pred = rr(video_pred, other_var)
















