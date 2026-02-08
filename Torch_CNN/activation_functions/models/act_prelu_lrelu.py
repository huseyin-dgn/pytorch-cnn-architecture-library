import torch
import torch.nn as nn

def _make_act(act: str, channels: int | None = None) -> nn.Module:
    a = act.lower()
    if a in ["lrelu", "leakyrelu", "leaky_relu"]:
        # slope sabit (istersen değiştir)
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if a == "prelu":
        # PReLU'nun parametre sayısı:
        # num_parameters=1 -> tek alpha (tüm kanallar için)
        # num_parameters=channels -> kanal başına alpha
        if channels is None:
            # güvenli fallback
            return nn.PReLU(num_parameters=1, init=0.25)
        return nn.PReLU(num_parameters=channels, init=0.25)

    raise ValueError("act must be 'prelu' or 'lrelu'/'leakyrelu'")

class DropPath(nn.Module):
    """Stochastic Depth (DropPath)."""
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
    def __init__(self, cin, cout, k=3, s=1, p=None, groups=1, act="lrelu"):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = _make_act(act, channels=cout)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SEGate(nn.Module):
    """SE-style channel gate (sigmoid)."""
    def __init__(self, channels: int, reduction: int = 16, act: str = "lrelu"):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = _make_act(act, channels=hidden)  # gate MLP içinde de aynı act
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x):
        w = self.pool(x)
        w = self.act(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w

class BottleneckSE(nn.Module):
    """Bottleneck residual + SE gate + DropPath"""
    def __init__(
        self,
        cin: int,
        cout: int,
        stride: int = 1,
        expansion: int = 4,
        reduction: int = 16,
        drop_path: float = 0.0,
        act: str = "lrelu",
    ):
        super().__init__()
        mid = cout // expansion
        assert mid > 0, "cout too small for given expansion"

        self.conv1 = ConvBNAct(cin, mid, k=1, s=1, p=0, act=act)
        self.conv2 = ConvBNAct(mid, mid, k=3, s=stride, p=1, act=act)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid, cout, kernel_size=1, bias=False),
            nn.BatchNorm2d(cout),
        )

        self.se = SEGate(cout, reduction=reduction, act=act)
        self.drop_path = DropPath(drop_path)

        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )

        self.out_act = _make_act(act, channels=cout)

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

def make_stage(cin, cout, num_blocks, stride, drop_path_rate, act):
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
                act=act,
            )
        )
    return nn.Sequential(*blocks)

class MidResNetSE(nn.Module):
    def __init__(self, num_classes=10, act="lrelu", drop_path_rate=0.10):
        super().__init__()
        self.act_name = act

        self.stem = nn.Sequential(
            ConvBNAct(3, 64, k=3, s=2, p=1, act=act),
            ConvBNAct(64, 64, k=3, s=1, p=1, act=act),
        )

        self.stage1 = make_stage(64, 128, num_blocks=2, stride=2, drop_path_rate=drop_path_rate, act=act)
        self.stage2 = make_stage(128, 256, num_blocks=3, stride=2, drop_path_rate=drop_path_rate, act=act)
        self.stage3 = make_stage(256, 512, num_blocks=2, stride=2, drop_path_rate=drop_path_rate, act=act)

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

def build_model(num_classes: int = 10, act: str = "lrelu") -> nn.Module:
    return MidResNetSE(num_classes=num_classes, act=act, drop_path_rate=0.10)

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    m_prelu = build_model(10, act="prelu")
    m_lrelu = build_model(10, act="lrelu")
    print("prelu out:", m_prelu(x).shape)
    print("lrelu out:", m_lrelu(x).shape)

######### Kullanım #########
# from models.act_prelu_lrelu import build_model

# model_prelu = build_model(num_classes=10, act="prelu")
# model_lrelu = build_model(num_classes=10, act="lrelu")