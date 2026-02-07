import torch
import torch.nn as nn

def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)

def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

def make_skip(in_ch, out_ch, stride):
    if stride == 1 and in_ch == out_ch:
        return nn.Identity()
    return nn.Sequential(
        conv1x1(in_ch, out_ch, stride),
        nn.BatchNorm2d(out_ch),
    )

class FxConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        f = self.act(self.bn1(self.conv1(x)))
        f = self.bn2(self.conv2(f))
        return f

class SEMask(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(self.pool(x))  # (B,C,1,1)

class NearIdentityGate(nn.Module):
    def __init__(self, mask:nn.Module,gamma_init = 0.0 , learn =True):
        super().__init__()
        self.mask = mask
        g = torch.tensor(float(gamma_init))
        if learn:
            self.gamma = nn.Parameter(g)
        else:
            self.register_buffer("gamma",g)
    
    def forward(self,x):
        m = self.mask(x)
        return x * (1.0 + self.gamma * m)
    
class Pattern4Block(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1,
        reduction_s=16,
        reduction_r=16,
        gamma_s_init=0.0,   # skip hafif/0
        gamma_r_init=0.1,   # residual biraz açık
    ):
        super().__init__()
        self.skip = make_skip(in_ch, out_ch, stride)
        self.F = FxConv(in_ch, out_ch, stride=stride)

        self.As = NearIdentityGate(SEMask(out_ch, reduction_s), gamma_init=gamma_s_init, learnable=True)
        self.Ar = NearIdentityGate(SEMask(out_ch, reduction_r), gamma_init=gamma_r_init, learnable=True)

        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        s = self.As(identity)

        f = self.F(x)
        r = self.Ar(f)

        y = s + r
        return self.out_act(y)

class BasicResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.skip = make_skip(in_ch, out_ch, stride)
        self.F = FxConv(in_ch, out_ch, stride=stride)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.skip(x) + self.F(x))


class AveragePattern4Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Stage-1 (erken katman): normal residual (stabil)
        self.stage1 = nn.Sequential(
            BasicResBlock(32, 64, stride=1),
            BasicResBlock(64, 64, stride=1),
        )
        # Stage-2 (orta): Pattern-4 başlıyor (skip hafif)
        self.stage2 = nn.Sequential(
            Pattern4Block(64, 128, stride=2, gamma_s_init=0.0, gamma_r_init=0.1),
            Pattern4Block(128, 128, stride=1, gamma_s_init=0.0, gamma_r_init=0.1),
        )
        # Stage-3 (daha semantik): Pattern-4 devam (istersen residual biraz daha güçlü)
        self.stage3 = nn.Sequential(
            Pattern4Block(128, 256, stride=2, gamma_s_init=0.0, gamma_r_init=0.15),
            Pattern4Block(256, 256, stride=1, gamma_s_init=0.0, gamma_r_init=0.15),
        )
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


if __name__ == "__main__":
    m = AveragePattern4Net(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    y = m(x)
    print("out:", y.shape)

    for n, mod in m.named_modules():
        if isinstance(mod, Pattern4Block):
            print(n, "gamma_s=", float(mod.As.gamma), "gamma_r=", float(mod.Ar.gamma))