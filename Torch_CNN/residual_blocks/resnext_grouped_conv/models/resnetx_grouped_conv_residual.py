import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_ch, out_ch, stride=1, groups=1):
    return nn.Conv2d(
        in_ch, out_ch, kernel_size=3, stride=stride, padding=1,
        groups=groups, bias=False
    )

class ResNeXtGroupedResidual(nn.Module):
    expansion = 4

    def __init__(self, in_ch, bottleneck_ch, stride=1, groups=4, base_width=4):
        super().__init__()

        # Çıkış kanalı (standart): out_ch = bottleneck_ch * 4
        out_ch = bottleneck_ch * self.expansion

        # Grouped 3x3'ün iç kanalı (width)
        # width = int(bottleneck_ch * (base_width/64)) * groups
        width = int(bottleneck_ch * (base_width / 64.0)) * groups

        # ---- Main path: F(x) ----
        self.conv1 = conv1x1(in_ch, width)      # kanal ayarla
        self.bn1   = nn.BatchNorm2d(width)

        self.conv2 = conv3x3(width, width, stride=stride, groups=groups)  # grouped 3x3
        self.bn2   = nn.BatchNorm2d(width)

        self.conv3 = conv1x1(width, out_ch)     # çıkış kanalına getir
        self.bn3   = nn.BatchNorm2d(out_ch)

        # ---- Skip path ----
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                conv1x1(in_ch, out_ch, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        # skip
        identity = x if self.downsample is None else self.downsample(x)

        # F(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        # y = F(x) + skip
        out = out + identity
        out = F.relu(out, inplace=True)
        return out