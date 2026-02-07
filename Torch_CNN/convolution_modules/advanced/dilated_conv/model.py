import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# Küçük yardımcı: Conv + BN + ReLU bloğu
# -------------------------------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -------------------------------------------------
# DilatedBlock:
# - 1. conv: normal 3x3 (d=1)
# - 2. conv: dilated 3x3 (d = dilation param)
# - Downsample gerekiyorsa stride=2 ile yapılır
# - Residual (shortcut) var
# -------------------------------------------------
class DilatedBlock(nn.Module):
    """
    c_in -> c_out

    Ana yol:
      conv1: 3x3, dilation=1
      conv2: 3x3, dilation=dilation

    Shortcut:
      - Eğer stride=1 ve c_in==c_out -> identity
      - Yoksa 1x1 conv (stride ile downsample)

    Bu blok ile:
      - Hem downsample yapabiliyorsun (stride=2 ile),
      - Hem de receptive field'i dilation ile büyütebiliyorsun.
    """
    def __init__(self, c_in, c_out, stride=1, dilation=1):
        super().__init__()

        # effective kernel: k_eff = d*(k-1)+1
        # k=3, dilation=d -> padding=d (H,W sabit kalsın diye, stride=1 iken)

        self.conv1 = ConvBNReLU(
            c_in=c_in,
            c_out=c_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=1,
        )

        self.conv2 = ConvBNReLU(
            c_in=c_out,
            c_out=c_out,
            kernel_size=3,
            stride=1,
            padding=dilation,   # k=3 için padding=dilation
            dilation=dilation,
        )

        # Shortcut
        if stride == 1 and c_in == c_out:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        shortcut = self.shortcut(x)
        out = out + shortcut
        out = F.relu(out, inplace=True)
        return out


# -------------------------------------------------
# DilatedNetSmall:
#  - Giriş: (N, 3, 32, 32)
#  - Stem: 3 -> 16
#  - Stage1: 2x DilatedBlock(16->16, d=1)
#  - Stage2: DilatedBlock(16->32, stride=2, d=2) + DilatedBlock(32->32, d=2)
#  - Stage3: DilatedBlock(32->64, stride=2, d=4) + DilatedBlock(64->64, d=4)
#  - GAP + FC
# -------------------------------------------------
class DilatedNetSmall(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Stem: normal conv, receptive field başlangıcı
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )  # (N, 16, 32, 32)

        # Stage 1: normal bloklar (dilation=1), boyut sabit
        self.stage1 = nn.Sequential(
            DilatedBlock(c_in=16, c_out=16, stride=1, dilation=1),  # (N,16,32,32)
            DilatedBlock(c_in=16, c_out=16, stride=1, dilation=1),  # (N,16,32,32)
        )

        # Stage 2: downsample + dilation=2
        self.stage2 = nn.Sequential(
            DilatedBlock(c_in=16, c_out=32, stride=2, dilation=2),  # (N,32,16,16)
            DilatedBlock(c_in=32, c_out=32, stride=1, dilation=2),  # (N,32,16,16)
        )

        # Stage 3: downsample + daha büyük dilation=4
        self.stage3 = nn.Sequential(
            DilatedBlock(c_in=32, c_out=64, stride=2, dilation=4),  # (N,64,8,8)
            DilatedBlock(c_in=64, c_out=64, stride=1, dilation=4),  # (N,64,8,8)
        )

        # Global Average Pool + FC
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # (N,64,1,1)
        self.fc = nn.Linear(64, num_classes)     # (N,64) -> (N,num_classes)

    def forward(self, x):
        # x: (N, 3, 32, 32)
        x = self.stem(x)      # (N, 16, 32, 32)
        x = self.stage1(x)    # (N, 16, 32, 32)
        x = self.stage2(x)    # (N, 32, 16, 16)
        x = self.stage3(x)    # (N, 64, 8, 8)
        x = self.gap(x)       # (N, 64, 1, 1)
        x = x.view(x.size(0), -1)  # (N, 64)
        x = self.fc(x)        # (N, num_classes)
        return x


# -------------------------------------------------
# Küçük test
# -------------------------------------------------
if __name__ == "__main__":
    model = DilatedNetSmall(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("Çıktı şekli:", y.shape)            # Beklenen: (4, 10)
    print("Toplam parametre:", count_params(model))
