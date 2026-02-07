import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# 1×1 Pointwise + 3×3 Conv ile Basit Bir Blok
# -------------------------------------------------
class ConvPointwiseBlock(nn.Module):
    def __init__(self, c_in, c_mid, c_out, stride=1):
        super().__init__()
        # 3x3 spatial conv
        self.conv3x3 = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_mid,
            kernel_size=3,
            stride=stride,       # stride ile downsample imkanı
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(c_mid)

        # 1x1 pointwise conv (channel mixing + C_mid -> C_out)
        self.pointwise = nn.Conv2d(
            in_channels=c_mid,
            out_channels=c_out,
            kernel_size=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.pointwise(x)   # kanal karışımı + kanal sayısını düzenleme
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        return x


# -------------------------------------------------
# Pointwise Ağırlıklı Küçük Bir CNN (CIFAR-10 için)
# -------------------------------------------------
class PointwiseNetSmall(nn.Module):
    """
    Giriş: (N, 3, 32, 32) — CIFAR-10 tarzı
    Yapı:
    - Stem: 3x3 Conv
    - Stage1: ConvPointwiseBlock x2 (boyut sabit, kanal 16 -> 32)
    - Stage2: ConvPointwiseBlock x2 (stride=2 ile 32 -> 16 spatial, kanal 32 -> 64)
    - Stage3: ConvPointwiseBlock x2 (stride=2 ile 16 -> 8 spatial, kanal 64 -> 128)
    - Global Average Pool + Linear
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # Stem: 3x3 Conv, sadece başlangıç
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # (N, 16, 32, 32)

        # Stage 1: spatial sabit, kanalları pointwise ile büyüt
        self.stage1 = nn.Sequential(
            ConvPointwiseBlock(c_in=16, c_mid=16, c_out=32, stride=1),
            ConvPointwiseBlock(c_in=32, c_mid=32, c_out=32, stride=1),
        )
        # (N, 32, 32, 32)

        # Stage 2: stride=2 ile downsample, kanal büyüt
        self.stage2 = nn.Sequential(
            ConvPointwiseBlock(c_in=32, c_mid=64, c_out=64, stride=2),
            ConvPointwiseBlock(c_in=64, c_mid=64, c_out=64, stride=1),
        )
        # (N, 64, 16, 16)

        # Stage 3: tekrar stride=2, kanal daha da büyüt
        self.stage3 = nn.Sequential(
            ConvPointwiseBlock(c_in=64, c_mid=128, c_out=128, stride=2),
            ConvPointwiseBlock(c_in=128, c_mid=128, c_out=128, stride=1),
        )
        # (N, 128, 8, 8)

        # Global Average Pool + Linear
        self.gap = nn.AdaptiveAvgPool2d((1, 1))   # (N, 128, 1, 1)
        self.fc  = nn.Linear(128, num_classes)    # (N, 128) -> (N, num_classes)

    def forward(self, x):
        # x: (N, 3, 32, 32)
        x = self.stem(x)      # (N, 16, 32, 32)
        x = self.stage1(x)    # (N, 32, 32, 32)
        x = self.stage2(x)    # (N, 64, 16, 16)
        x = self.stage3(x)    # (N, 128, 8, 8)
        x = self.gap(x)       # (N, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (N, 128)
        x = self.fc(x)        # (N, num_classes)
        return x


# -------------------------------------------------
# Küçük Test
# -------------------------------------------------
if __name__ == "__main__":
    model = PointwiseNetSmall(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("Çıktı şekli:", y.shape)              # Beklenen: (4, 10)
    print("Toplam parametre:", count_params(model))