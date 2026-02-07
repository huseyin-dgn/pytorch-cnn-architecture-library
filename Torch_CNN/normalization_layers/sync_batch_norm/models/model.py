# model_syncbn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)  # sonra SyncBN'e çevireceğiz
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.f1 = ConvBNAct(ch, ch, k=3, s=1, p=1)
        self.f2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.f2(self.f1(x)))

class SyncBNNet(nn.Module):
    def __init__(self, num_classes=10, width=32):
        super().__init__()
        w = width

        self.stem = ConvBNAct(3, w, k=3, s=2, p=1)         # 1/2
        self.stage1 = nn.Sequential(
            ConvBNAct(w, w*2, k=3, s=2, p=1),              # 1/4
            ResidualBlock(w*2),
            ResidualBlock(w*2),
        )
        self.stage2 = nn.Sequential(
            ConvBNAct(w*2, w*4, k=3, s=2, p=1),            # 1/8
            ResidualBlock(w*4),
            ResidualBlock(w*4),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(w*4, w*8, k=3, s=2, p=1),            # 1/16
            ResidualBlock(w*8),
            ResidualBlock(w*8),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(w*8, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    m = SyncBNNet(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print(y.shape)