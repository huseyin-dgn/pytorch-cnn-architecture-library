import torch
import torch.nn as nn
import torch.nn.functional as F

class SeBlock(nn.Module):
    def __init__(self, channels, reductions=8):
        super().__init__()
        hidden = max(1, channels // reductions)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        s = self.pool(x)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s

def cords(B, H, W, device, dtype, add_rad=False):
    y = torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    xx = xx[None, None].repeat(B, 1, 1, 1)
    yy = yy[None, None].repeat(B, 1, 1, 1)

    if add_rad:
        rr = torch.sqrt(xx**2 + yy**2)
        return torch.cat([xx, yy, rr], dim=1)
    return torch.cat([xx, yy], dim=1)

class CordConv(nn.Module):
    def __init__(self, cin, cout, k=3, stride=1, padding=1, add_rad=False, bias=False):
        super().__init__()
        self.add_rad = add_rad
        extras = 3 if add_rad else 2
        self.conv = nn.Conv2d(cin + extras, cout, kernel_size=k, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        coords = cords(b, h, w, device=x.device, dtype=x.dtype, add_rad=self.add_rad)
        x = torch.cat([x, coords], dim=1)
        return self.conv(x)

class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class CoordSEHeadXYNet(nn.Module):
    """
    CoordConv (head) + SE ile (x,y) regresyon modeli
    Çıkış: (B,2) in [-1,1]
    """
    def __init__(self, in_channels=3, base=32, add_rad=True, se_reduction=8):
        super().__init__()

        c1, c2, c3 = base, base*2, base*4

        self.backbone = nn.Sequential(
            ConvBNReLU(in_channels, c1, 3, 1, 1),
            nn.MaxPool2d(2),
            ConvBNReLU(c1, c2, 3, 2, 1),
            ConvBNReLU(c2, c3, 3, 2, 1),
        )

        self.head = nn.Sequential(
            CordConv(c3, c3, k=3, stride=1, padding=1, add_rad=add_rad, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            SeBlock(c3, reductions=se_reduction),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c3, 2)

    def forward(self, x, verbose=False):
        if verbose: print("in      :", x.shape)
        f = self.backbone(x)
        if verbose: print("features :", f.shape)
        h = self.head(f)
        if verbose: print("head     :", h.shape)
        h = self.pool(h)
        h = torch.flatten(h, 1)
        xy = torch.tanh(self.fc(h))
        if verbose: print("xy       :", xy.shape)
        return xy

# hızlı test
if __name__ == "__main__":
    model = CoordSEHeadXYNet(in_channels=3, base=32, add_rad=True, se_reduction=8)
    x = torch.randn(8, 3, 64, 64)
    y = model(x, verbose=True)
    print("y:", y.shape)