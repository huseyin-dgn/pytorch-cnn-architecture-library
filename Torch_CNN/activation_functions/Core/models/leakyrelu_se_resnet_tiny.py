import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, min_hidden: int = 4):
        super().__init__()
        hidden = max(min_hidden, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class ConvBNAct(nn.Module):
    """Conv -> BN -> LeakyReLU"""
    def __init__(self, cin, cout, k=3, s=1, p=1, groups=1, negative_slope=0.1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualSEBlock(nn.Module):
    def __init__(self, channels: int, negative_slope=0.1):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, k=3, s=1, p=1, negative_slope=negative_slope)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se = SEBlock(channels)
        self.act_out = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = out + x
        return self.act_out(out)


class TinySE_ResNet_LeakyReLU(nn.Module):
    def __init__(self, num_classes: int = 10, negative_slope: float = 0.1):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNAct(3, 32, s=2, negative_slope=negative_slope),
            ConvBNAct(32, 32, negative_slope=negative_slope),
        )

        self.stage1 = nn.Sequential(
            ConvBNAct(32, 64, s=2, negative_slope=negative_slope),
            ResidualSEBlock(64, negative_slope=negative_slope),
            ResidualSEBlock(64, negative_slope=negative_slope),
        )

        self.stage2 = nn.Sequential(
            ConvBNAct(64, 128, s=2, negative_slope=negative_slope),
            ResidualSEBlock(128, negative_slope=negative_slope),
            ResidualSEBlock(128, negative_slope=negative_slope),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
