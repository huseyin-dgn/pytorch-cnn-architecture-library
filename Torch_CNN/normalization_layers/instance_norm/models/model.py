import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvINAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act="relu", affine=True):
        super().__init__()
        if p is None:
            p = k // 2

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)

        self.inorm = nn.InstanceNorm2d(out_ch, eps=1e-5, affine=affine, track_running_stats=False)

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "gelu":
            self.act = nn.GELU()
        elif act in ("none", None):
            self.act = nn.Identity()
        else:
            raise ValueError("act: 'relu'|'silu'|'gelu'|'none'")

    def forward(self, x):
        return self.act(self.inorm(self.conv(x)))


class SimpleIN_AutoEncoder(nn.Module):
    def __init__(self, base=32, in_affine=True):
        super().__init__()

        # Encoder (H,W -> H/4,W/4)
        self.enc1 = nn.Sequential(
            ConvINAct(3, base,   k=7, s=1, p=3, act="relu", affine=in_affine),
            ConvINAct(base, base, k=3, s=2, act="relu", affine=in_affine),      # /2
        )
        self.enc2 = nn.Sequential(
            ConvINAct(base, base*2, k=3, s=2, act="relu", affine=in_affine),    # /4
            ConvINAct(base*2, base*2, k=3, s=1, act="relu", affine=in_affine),
        )

        # Bottleneck
        self.mid = nn.Sequential(
            ConvINAct(base*2, base*2, k=3, s=1, act="relu", affine=in_affine),
            ConvINAct(base*2, base*2, k=3, s=1, act="relu", affine=in_affine),
        )

        # Decoder (H/4 -> H)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvINAct(base*2, base, k=3, s=1, act="relu", affine=in_affine),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvINAct(base, base, k=3, s=1, act="relu", affine=in_affine),
        )
        self.out = nn.Conv2d(base, 3, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.mid(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    model = SimpleIN_AutoEncoder(base=32, in_affine=True)
    inp = torch.randn(2, 3, 128, 128)
    out = model(inp)
    print("inp:", inp.shape, "out:", out.shape)