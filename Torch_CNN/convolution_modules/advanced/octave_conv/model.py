import torch
import torch.nn as nn
import torch.nn.functional as F

class OctaveConv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, stride=1, padding=1,
                 alpha_in=0.5, alpha_out=0.5, bias=False):
        super().__init__()
        cin_l = int(round(cin * alpha_in)); cin_h = cin - cin_l
        cout_l = int(round(cout * alpha_out)); cout_h = cout - cout_l
        self.cin_h, self.cin_l = cin_h, cin_l
        self.cout_h, self.cout_l = cout_h, cout_l

        self.down = nn.AvgPool2d(2, 2)

        self.hh = nn.Conv2d(cin_h, cout_h, kernel_size, stride=stride, padding=padding, bias=bias) if (cin_h and cout_h) else None
        self.hl = nn.Conv2d(cin_h, cout_l, kernel_size, stride=stride, padding=padding, bias=bias) if (cin_h and cout_l) else None
        self.ll = nn.Conv2d(cin_l, cout_l, kernel_size, stride=stride, padding=padding, bias=bias) if (cin_l and cout_l) else None
        self.lh = nn.Conv2d(cin_l, cout_h, kernel_size, stride=stride, padding=padding, bias=bias) if (cin_l and cout_h) else None

    def forward(self, x_h, x_l=None):
        y_h = 0
        y_l = 0

        if self.hh is not None and x_h is not None:
            y_h = y_h + self.hh(x_h)

        if self.hl is not None and x_h is not None:
            y_l = y_l + self.hl(self.down(x_h))

        if self.ll is not None and x_l is not None:
            y_l = y_l + self.ll(x_l)

        if self.lh is not None and x_l is not None:
            y_lh = self.lh(x_l)
            y_h = y_h + F.interpolate(y_lh, scale_factor=2, mode="nearest")

        if isinstance(y_h, int): y_h = None
        if isinstance(y_l, int): y_l = None
        return y_h, y_l


class OctaveConvBlock(nn.Module):
    def __init__(self, cin, cout, alpha_in=0.5, alpha_out=0.5,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.oct = OctaveConv2d(cin, cout, kernel_size, stride, padding, alpha_in, alpha_out, bias=False)
        self.bn_h = nn.BatchNorm2d(self.oct.cout_h) if self.oct.cout_h > 0 else None
        self.bn_l = nn.BatchNorm2d(self.oct.cout_l) if self.oct.cout_l > 0 else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x_h, x_l=None):
        y_h, y_l = self.oct(x_h, x_l)
        if y_h is not None and self.bn_h is not None:
            y_h = self.act(self.bn_h(y_h))
        if y_l is not None and self.bn_l is not None:
            y_l = self.act(self.bn_l(y_l))
        return y_h, y_l


class SplitMergeHL(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.down = nn.AvgPool2d(2, 2)

    def split(self, x):
        # x: (B, C, H, W) -> x_h: (B, C_h, H, W), x_l: (B, C_l, H/2, W/2)
        B, C, H, W = x.shape
        c_l = int(round(C * self.alpha))
        c_h = C - c_l
        x_h = x[:, :c_h]
        x_l = self.down(x[:, c_h:]) if c_l > 0 else None
        return x_h, x_l

    def merge(self, x_h, x_l):
        # -> (B, C_h + C_l, H, W)
        if x_l is None:
            return x_h
        x_l_up = F.interpolate(x_l, scale_factor=2, mode="nearest")
        return torch.cat([x_h, x_l_up], dim=1)


class OctNetMini(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, alpha=0.5):
        super().__init__()
        self.sm = SplitMergeHL(alpha=alpha)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.b1 = OctaveConvBlock(64, 128, alpha_in=alpha, alpha_out=alpha)
        self.b2 = OctaveConvBlock(128, 128, alpha_in=alpha, alpha_out=alpha)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x_h, x_l = self.sm.split(x)
        x_h, x_l = self.b1(x_h, x_l)
        x_h, x_l = self.b2(x_h, x_l)
        x = self.sm.merge(x_h, x_l)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


if __name__ == "__main__":
    model = OctNetMini(in_channels=3, num_classes=10, alpha=0.5)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print("Input :", x.shape)
    print("Output:", y.shape)  # (4, 10)
