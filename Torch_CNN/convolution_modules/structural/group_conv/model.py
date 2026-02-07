import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# 1) Basit blok: Conv -> BN -> ReLU
# ------------------------------------------------------------
class CBR(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p,
                              groups=groups, bias=False)  # groups=G => group conv
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ------------------------------------------------------------
# 2) GroupConv’lu Residual Block (ResNeXt-style)
#    1x1 reduce -> 3x3 group conv -> 1x1 expand + skip
# ------------------------------------------------------------
class GroupResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, groups=8, bottleneck_ratio=0.5):
        super().__init__()

        mid = int(cout * bottleneck_ratio)          # ara kanal (bottleneck)
        mid = max(1, mid)

        # mid % groups şart (group conv çalışması için)
        if mid % groups != 0:
            # en yakın uygun grubu bul (okunabilir olsun diye basit çözüm)
            for g in range(groups, 0, -1):
                if mid % g == 0:
                    groups = g
                    break
        self.groups = groups

        self.reduce = CBR(cin, mid, k=1, s=1, p=0)                 # kanal düşür + mixing
        self.gconv  = CBR(mid, mid, k=3, s=stride, p=1, groups=groups)  # asıl group conv
        self.expand = nn.Sequential(                                # kanal büyüt
            nn.Conv2d(mid, cout, kernel_size=1, bias=False),
            nn.BatchNorm2d(cout)
        )

        # skip bağlantısı: shape uyuşmazsa 1x1 ile eşle
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(cout)
            )
        else:
            self.skip = nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)     # residual yol
        out = self.reduce(x)        # 1x1
        out = self.gconv(out)       # 3x3 group
        out = self.expand(out)      # 1x1
        out = out + identity        # residual toplama
        return self.act(out)


# ------------------------------------------------------------
# 3) Tam Model: Stem + 3 Stage + Head
# ------------------------------------------------------------
class GroupConvNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, groups=8, blocks=(2, 2, 2)):
        super().__init__()

        # Stem: erken aşamada normal conv (mixing güçlü olsun)
        self.stem = nn.Sequential(
            CBR(in_channels, 64, k=3, s=1, p=1, groups=1),
            CBR(64, 64, k=3, s=1, p=1, groups=1),
        )

        # Stage'ler: ilk blok stride=2 ile downsample yapar
        self.stage1 = self._make_stage(64, 128, n=blocks[0], stride=2, groups=groups)
        self.stage2 = self._make_stage(128, 256, n=blocks[1], stride=2, groups=groups)
        self.stage3 = self._make_stage(256, 512, n=blocks[2], stride=2, groups=groups)

        # Head: global pooling + linear
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_stage(self, cin, cout, n, stride, groups):
        layers = []
        layers.append(GroupResBlock(cin, cout, stride=stride, groups=groups))  # downsample blok
        for _ in range(n - 1):
            layers.append(GroupResBlock(cout, cout, stride=1, groups=groups))  # normal bloklar
        return nn.Sequential(*layers)

    def forward(self, x, verbose=False):
        if verbose: print("input  :", x.shape)

        x = self.stem(x)
        if verbose: print("stem   :", x.shape)

        x = self.stage1(x)
        if verbose: print("stage1 :", x.shape)

        x = self.stage2(x)
        if verbose: print("stage2 :", x.shape)

        x = self.stage3(x)
        if verbose: print("stage3 :", x.shape)

        x = self.pool(x)
        if verbose: print("pool   :", x.shape)

        x = torch.flatten(x, 1)
        if verbose: print("flat   :", x.shape)

        x = self.fc(x)
        if verbose: print("logits :", x.shape)

        return x


# ------------------------------------------------------------
# Çalıştırma
# ------------------------------------------------------------
if __name__ == "__main__":
    model = GroupConvNet(in_channels=3, num_classes=10, groups=8, blocks=(2, 2, 2))
    x = torch.randn(2, 3, 128, 128)
    y = model(x, verbose=True)
    print("\nfinal:", y.shape)
