import torch
import torch.nn as nn
import torch.nn.functional as F

class HardSwish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.hardswish(x)

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=None, groups=1, act=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = HardSwish()  # <-- HARDSWISH

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SEGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)
        self.act = HardSwish()

    def forward(self, x):
        w = self.pool(x)
        w = self.act(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w

class BottleneckSE(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        stride: int = 1,
        expansion: int = 4,
        reduction: int = 16,
        drop_path: float = 0.0,
    ):
        super().__init__()
        mid = cout // expansion
        assert mid > 0, "cout too small for given expansion"

        self.conv1 = ConvBNAct(cin, mid, k=1, s=1, p=0)
        self.conv2 = ConvBNAct(mid, mid, k=3, s=stride, p=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid, cout, kernel_size=1, bias=False),
            nn.BatchNorm2d(cout),
        )

        self.se = SEGate(cout, reduction=reduction)
        self.drop_path = DropPath(drop_path)

        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )

        self.out_act = HardSwish()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.se(out)
        out = self.drop_path(out)

        if self.down is not None:
            identity = self.down(identity)

        out = out + identity
        return self.out_act(out)

def make_stage(cin, cout, num_blocks, stride, drop_path_rate):
    blocks = []
    for i in range(num_blocks):
        s = stride if i == 0 else 1
        blocks.append(
            BottleneckSE(
                cin=cin if i == 0 else cout,
                cout=cout,
                stride=s,
                expansion=4,
                reduction=16,
                drop_path=drop_path_rate,
            )
        )
    return nn.Sequential(*blocks)

class MidResNetSE(nn.Module):
    def __init__(self, num_classes=10, drop_path_rate=0.10):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 64, k=3, s=2, p=1),
            ConvBNAct(64, 64, k=3, s=1, p=1),
        )

        self.stage1 = make_stage(64, 128, num_blocks=2, stride=2, drop_path_rate=drop_path_rate)
        self.stage2 = make_stage(128, 256, num_blocks=3, stride=2, drop_path_rate=drop_path_rate)
        self.stage3 = make_stage(256, 512, num_blocks=2, stride=2, drop_path_rate=drop_path_rate)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.head(x)


def build_model(num_classes: int = 10) -> nn.Module:
    return MidResNetSE(num_classes=num_classes, drop_path_rate=0.10)