import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialDropout2d(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError("p 0 ve 1 arasında olmalı")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (self.p == 0.0):
            return x
        q = 1.0 - self.p
        mask = torch.empty((x.size(0), x.size(1), 1, 1), device=x.device, dtype=x.dtype).bernoulli_(q)
        return x * mask / q


class ConvBNActSD(nn.Module):
    """Conv -> BN -> Act -> SpatialDropout2d"""
    def __init__(self, cin: int, cout: int, k=3, s=1, p=1, sd_p: float = 0.1, act: str = "silu"):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True)
        self.sd = SpatialDropout2d(sd_p)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.sd(x)
        return x


class BasicResBlockSD(nn.Module):
    """(Conv->BN->Act->Conv->BN->SD) + skip -> Act"""
    def __init__(self, cin: int, cout: int, stride: int = 1, sd_p: float = 0.1, act: str = "relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

        self.sd = SpatialDropout2d(sd_p)

        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.sd(out)           # Spatial Dropout branch içinde
        out = self.act(out + identity)
        return out


class TinyCNNSpatialDropout(nn.Module):
    """
    Spatial Dropout kullanan örnek CNN:
    - Stem: ConvBNActSD
    - Downsample
    - 2 Residual block (SD'li)
    - GAP + FC
    """
    def __init__(self, num_classes: int = 10, sd_p: float = 0.1):
        super().__init__()
        self.stem = ConvBNActSD(3, 32, k=3, s=1, p=1, sd_p=sd_p, act="silu")

        self.down = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )

        self.block1 = BasicResBlockSD(64, 64, stride=1, sd_p=sd_p, act="relu")
        self.block2 = BasicResBlockSD(64, 64, stride=1, sd_p=sd_p, act="relu")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.down(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# quick test
if __name__ == "__main__":
    model = TinyCNNSpatialDropout(num_classes=5, sd_p=0.2)
    model.train()  # dropout aktif
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    print(y.shape)  # [2, 5]