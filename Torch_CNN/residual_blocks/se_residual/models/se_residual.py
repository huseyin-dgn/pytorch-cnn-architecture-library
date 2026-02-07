import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# SE (Squeeze-and-Excitation)
# =========================
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


# =========================
# SE Residual Block
# =========================
class SEResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, reduction: int = 16):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.se = SEBlock(out_ch, reduction)

        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)

        f = F.relu(self.bn1(self.conv1(x)), inplace=True)
        f = self.bn2(self.conv2(f))
        f = self.se(f)

        out = F.relu(f + identity, inplace=True)
        return out


# =========================
# SE-ResNet Mini Model
# =========================
class SEResNetMini(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            SEResidualBlock(32, 32, stride=1),
            SEResidualBlock(32, 32, stride=1),
        )

        self.stage2 = nn.Sequential(
            SEResidualBlock(32, 64, stride=2),
            SEResidualBlock(64, 64, stride=1),
        )

        self.stage3 = nn.Sequential(
            SEResidualBlock(64, 128, stride=2),
            SEResidualBlock(128, 128, stride=1),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


# =========================
# Run Test
# =========================
if __name__ == "__main__":
    model = SEResNetMini(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)  # [4, 10]
