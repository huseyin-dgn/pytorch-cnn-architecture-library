import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# MBConv Bloğu
# ----------------------
class MBConv(nn.Module):
    """
    Inverted Bottleneck (MBConv) bloğu:
    1x1 expand  -> 3x3 depthwise -> 1x1 project (+ residual)
    """
    def __init__(self, c_in, c_out, expansion_factor=4, stride=1, kernel_size=3):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and c_in == c_out)

        c_exp = c_in * expansion_factor

        # 1) 1x1 Expansion
        self.expand = nn.Sequential(
            nn.Conv2d(c_in, c_exp, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_exp),
            nn.SiLU(inplace=True)
        )

        # 2) 3x3 Depthwise
        padding = kernel_size // 2
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                c_exp,
                c_exp,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=c_exp,   # depthwise
                bias=False
            ),
            nn.BatchNorm2d(c_exp),
            nn.SiLU(inplace=True)
        )

        # 3) 1x1 Projection (linear)
        self.project = nn.Sequential(
            nn.Conv2d(c_exp, c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out)
            # aktivasyon yok (linear bottleneck)
        )

    def forward(self, x):
        identity = x

        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)

        if self.use_residual:
            out = out + identity

        return out


# ----------------------
# MBConv Entegre Küçük Model
# CIFAR-10 (3x32x32) için
# ----------------------
class MBConvNetSmall(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Stem: ilk konvolüsyon (hafif)
        # (N, 3, 32, 32) -> (N, 16, 32, 32)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True)
        )

        # Stage 1 (boyut sabit, kanallar 16)
        # (N, 16, 32, 32) -> (N, 16, 32, 32)
        self.stage1 = nn.Sequential(
            MBConv(16, 16, expansion_factor=4, stride=1),
            MBConv(16, 16, expansion_factor=4, stride=1),
        )

        # Stage 2 (downsample + kanal artırma)
        # (N, 16, 32, 32) -> (N, 24, 16, 16)
        self.stage2 = nn.Sequential(
            MBConv(16, 24, expansion_factor=4, stride=2),
            MBConv(24, 24, expansion_factor=4, stride=1),
        )

        # Stage 3 (downsample + kanal artırma)
        # (N, 24, 16, 16) -> (N, 40, 8, 8)
        self.stage3 = nn.Sequential(
            MBConv(24, 40, expansion_factor=4, stride=2),
            MBConv(40, 40, expansion_factor=4, stride=1),
        )

        # Global average pooling + FC
        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # (N, 40, 8, 8) -> (N, 40, 1, 1)
        self.fc   = nn.Linear(40, num_classes)

    def forward(self, x):
        # x: (N, 3, 32, 32)
        x = self.stem(x)     # (N, 16, 32, 32)
        x = self.stage1(x)   # (N, 16, 32, 32)
        x = self.stage2(x)   # (N, 24, 16, 16)
        x = self.stage3(x)   # (N, 40, 8, 8)
        x = self.pool(x)     # (N, 40, 1, 1)
        x = x.view(x.size(0), -1)  # (N, 40)
        x = self.fc(x)       # (N, num_classes)
        return x


# ----------------------
# Küçük test
# ----------------------
if __name__ == "__main__":
    model = MBConvNetSmall(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("Çıktı şekli:", y.shape)              # Beklenen: (4, 10)
    print("Toplam parametre:", count_params(model))
