import torch
import torch.nn as nn
import torch.nn.functional as F

# Depthwise + Pointwise blok
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # 1) Depthwise: her kanal kendi filtresiyle (groups=c_in)
        self.depthwise = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=c_in,
            bias=False
        )
        # 2) Pointwise: 1x1 ile kanal karışımı + kanal sayısını değiştirme
        self.pointwise = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# Sadece DepthwiseSeparableConv kullanan mini model
class TinyDepthwiseNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Giriş: (N, 3, 32, 32) varsayıyoruz (CIFAR-10 gibi)
        self.block1 = DepthwiseSeparableConv(3, 32, kernel_size=3, padding=1)
        self.block2 = DepthwiseSeparableConv(32, 64, kernel_size=3, padding=1)
        self.block3 = DepthwiseSeparableConv(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)  # 32 -> 16 -> 8
        self.gap  = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pool
        self.fc   = nn.Linear(128, num_classes)

    def forward(self, x):
        # 32x32
        x = F.relu(self.block1(x))          # (N, 32, 32, 32)
        x = self.pool(F.relu(self.block2(x)))  # (N, 64, 16, 16)
        x = self.pool(F.relu(self.block3(x)))  # (N, 128, 8, 8)

        x = self.gap(x)                     # (N, 128, 1, 1)
        x = x.view(x.size(0), -1)           # (N, 128)
        x = self.fc(x)                      # (N, num_classes)
        return x


# Küçük test
if __name__ == "__main__":
    model = TinyDepthwiseNet(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("Çıktı şekli:", y.shape)              # Beklenen: (4, 10)
    print("Toplam parametre:", count_params(model))
