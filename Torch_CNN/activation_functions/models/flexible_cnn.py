import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Activation Factory
# ---------------------------
def make_act(name: str, channels: int | None = None):
    n = name.lower()
    if n == "relu":
        return nn.ReLU(inplace=True)
    if n in ["lrelu", "leakyrelu"]:
        return nn.LeakyReLU(0.1, inplace=True)
    if n == "prelu":
        return nn.PReLU(num_parameters=(channels or 1), init=0.25)
    if n in ["silu", "swish"]:
        return nn.SiLU(inplace=True)
    if n == "hardswish":
        return nn.Hardswish(inplace=True)
    raise ValueError(f"Unknown activation: {name}")

def make_gate(name: str):
    g = name.lower()
    if g == "sigmoid":
        return torch.sigmoid
    if g == "hardsigmoid":
        return F.hardsigmoid
    raise ValueError(f"Unknown gate: {name}")

# ---------------------------
# Basic Blocks
# ---------------------------
class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=None, act="silu"):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(cin, cout, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = make_act(act, cout)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ---------------------------
# Attention Blocks
# ---------------------------
class SE(nn.Module):
    def __init__(self, c, r=16, act="silu", gate="sigmoid"):
        super().__init__()
        h = max(4, c // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, h, 1)
        self.act = make_act(act, h)
        self.fc2 = nn.Conv2d(h, c, 1)
        self.gate = make_gate(gate)

    def forward(self, x):
        w = self.pool(x)
        w = self.act(self.fc1(w))
        w = self.gate(self.fc2(w))
        return x * w

class CBAM(nn.Module):
    def __init__(self, c, r=16, act="silu", gate="sigmoid"):
        super().__init__()
        self.se = SE(c, r, act, gate)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        x = self.se(x)
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)
        s = torch.sigmoid(self.spatial(s))
        return x * s

class CoordAtt(nn.Module):
    def __init__(self, c, r=32, act="silu", gate="sigmoid"):
        super().__init__()
        h = max(8, c // r)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(c, h, 1)
        self.act = make_act(act, h)
        self.conv_h = nn.Conv2d(h, c, 1)
        self.conv_w = nn.Conv2d(h, c, 1)
        self.gate = make_gate(gate)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0,1,3,2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.conv1(y))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0,1,3,2)
        a_h = self.gate(self.conv_h(y_h))
        a_w = self.gate(self.conv_w(y_w))
        return x * a_h * a_w

def make_attention(name: str, c: int, act="silu", gate="sigmoid"):
    n = name.lower()
    if n == "none":
        return nn.Identity()
    if n == "se":
        return SE(c, act=act, gate=gate)
    if n == "cbam":
        return CBAM(c, act=act, gate=gate)
    if n in ["coord", "coordinate"]:
        return CoordAtt(c, act=act, gate=gate)
    raise ValueError(f"Unknown attention: {name}")

# ---------------------------
# Residual Block
# ---------------------------
class ResBlock(nn.Module):
    def __init__(self, cin, cout, s=1, act="silu", attn="none", gate="sigmoid"):
        super().__init__()
        self.conv1 = ConvBNAct(cin, cout, 3, s, act=act)
        self.conv2 = ConvBNAct(cout, cout, 3, 1, act=act)
        self.attn = make_attention(attn, cout, act=act, gate=gate)
        self.down = None
        if s != 1 or cin != cout:
            self.down = nn.Sequential(
                nn.Conv2d(cin, cout, 1, s, bias=False),
                nn.BatchNorm2d(cout)
            )
        self.out_act = make_act(act, cout)

    def forward(self, x):
        idt = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attn(x)
        if self.down is not None:
            idt = self.down(idt)
        return self.out_act(x + idt)

# ---------------------------
# Flexible CNN
# ---------------------------
class FlexibleCNN(nn.Module):
    """
    activation: relu | lrelu | prelu | silu | swish | hardswish
    attention: none | se | cbam | coord
    gate: sigmoid | hardsigmoid
    """
    def __init__(self, num_classes=10, activation="silu", attention="none", gate="sigmoid"):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 64, 3, 2, act=activation),
            ConvBNAct(64, 64, 3, 1, act=activation),
        )
        self.stage1 = ResBlock(64, 128, 2, act=activation, attn=attention, gate=gate)
        self.stage2 = ResBlock(128, 256, 2, act=activation, attn=attention, gate=gate)
        self.stage3 = ResBlock(256, 512, 2, act=activation, attn=attention, gate=gate)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

def build_model(num_classes=10, activation="silu", attention="none", gate="sigmoid"):
    return FlexibleCNN(num_classes, activation, attention, gate)

if __name__ == "__main__":
    x = torch.randn(2,3,224,224)
    m = build_model(
        num_classes=10,
        activation="prelu",
        attention="se",
        gate="hardsigmoid"
    )
    y = m(x)
    print(y.shape)
