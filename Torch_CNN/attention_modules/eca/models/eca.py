import math
import torch
import torch.nn as nn

def _make_odd(k: int) -> int:
    return k if (k % 2 == 1) else (k + 1)

def eca_kernel_size(channels: int, gamma: int = 2, b: int = 1, k_min: int = 1, k_max: int = 15) -> int:
    k = int(abs((math.log2(max(1, channels)) / gamma) + b))
    k = _make_odd(max(k_min, min(k, k_max)))
    return k

class ECABlock(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        k = eca_kernel_size(channels, gamma=gamma, b=b)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1)//2, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x).squeeze(-1).squeeze(-1).unsqueeze(1)
        a = self.conv1d(s)
        w = self.gate(a).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        return x * w

class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, act="silu"):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ECAResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, gamma=2, b=1):
        super().__init__()
        self.conv1 = ConvBNAct(cin, cout, k=3, s=stride, p=1, act="silu")
        self.conv2 = nn.Sequential(
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.eca = ECABlock(cout, gamma=gamma, b=b)
        self.out_act = nn.SiLU(inplace=True)

        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout)
            )

    def forward(self, x):
        identity = x if self.short is None else self.short(x)
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.eca(y)
        y = y + identity
        return self.out_act(y)

class ECAResNetClassifier(nn.Module):
    def __init__(self, num_classes=10, in_ch=3, base=32, gamma=2, b=1, dropout=0.0):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, base, k=3, s=2, p=1),
            ConvBNAct(base, base, k=3, s=1, p=1),
        )

        self.stage1 = nn.Sequential(
            ECAResBlock(base, base, stride=1, gamma=gamma, b=b),
            ECAResBlock(base, base, stride=1, gamma=gamma, b=b),
        )
        self.stage2 = nn.Sequential(
            ECAResBlock(base, base * 2, stride=2, gamma=gamma, b=b),
            ECAResBlock(base * 2, base * 2, stride=1, gamma=gamma, b=b),
        )
        self.stage3 = nn.Sequential(
            ECAResBlock(base * 2, base * 4, stride=2, gamma=gamma, b=b),
            ECAResBlock(base * 4, base * 4, stride=1, gamma=gamma, b=b),
        )
        self.stage4 = nn.Sequential(
            ECAResBlock(base * 4, base * 8, stride=2, gamma=gamma, b=b),
            ECAResBlock(base * 8, base * 8, stride=1, gamma=gamma, b=b),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(base * 8, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)

if __name__ == "__main__":
    m = ECAResNetClassifier(num_classes=10, in_ch=3, base=32, dropout=0.1)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print(y.shape)
