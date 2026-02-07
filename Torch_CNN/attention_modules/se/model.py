import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, min_hidden: int = 4):
        super().__init__()
        hidden = max(min_hidden, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w

class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, act="silu"):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SEResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, reduction=16):
        super().__init__()
        self.conv1 = ConvBNAct(cin, cout, k=3, s=stride, p=1, act="silu")
        self.conv2 = nn.Sequential(
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout)
        )
        self.se = SEBlock(cout, reduction=reduction)

        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout)
            )

        self.out_act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x if self.short is None else self.short(x)
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.se(y)
        y = y + identity
        return self.out_act(y)

class SEResNetLite(nn.Module):
    def __init__(self, num_classes=10, in_ch=3, base=32, reduction=16):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, base, k=3, s=2, p=1),
            ConvBNAct(base, base, k=3, s=1, p=1),
        )

        self.stage1 = nn.Sequential(
            SEResBlock(base, base, stride=1, reduction=reduction),
            SEResBlock(base, base, stride=1, reduction=reduction),
        )
        self.stage2 = nn.Sequential(
            SEResBlock(base, base * 2, stride=2, reduction=reduction),
            SEResBlock(base * 2, base * 2, stride=1, reduction=reduction),
        )
        self.stage3 = nn.Sequential(
            SEResBlock(base * 2, base * 4, stride=2, reduction=reduction),
            SEResBlock(base * 4, base * 4, stride=1, reduction=reduction),
        )
        self.stage4 = nn.Sequential(
            SEResBlock(base * 4, base * 8, stride=2, reduction=reduction),
            SEResBlock(base * 8, base * 8, stride=1, reduction=reduction),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base * 8, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    m = SEResNetLite(num_classes=10, in_ch=3, base=32, reduction=16)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print(y.shape)
