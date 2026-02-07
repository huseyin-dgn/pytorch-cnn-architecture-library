import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 1) Weight Standardized Convolution
# =========================================================
class WSConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight  # <-- WS: burada ham weight'i alıyoruz (optimizer bunu öğrenir)

        w = w - w.mean(dim=(1, 2, 3), keepdim=True)  # <-- WS: her out_channel için mean çıkar (merkezle)

        w = w / (w.flatten(1).std(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-5)  
        # <-- WS: her out_channel için std ile böl (ölçeği 1'e çek) + eps (0'a bölmeyi engelle)

        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # <-- WS: Conv hesaplamasını standardize edilmiş weight (w) ile yapıyoruz


# =========================================================
# 2) WS + GN + Aktivasyon Bloğu
# =========================================================
class WSGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=32):
        super().__init__()
        self.conv = WSConv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)  
        # <-- WS: Burada normal Conv2d yerine WSConv2d kullanıyoruz (WS uygulanan nokta)
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class WSResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, groups=32):
        super().__init__()
        self.conv1 = WSGNAct(in_ch, out_ch, k=3, s=stride, p=1, groups=groups)
        self.conv2 = WSGNAct(out_ch, out_ch, k=3, s=1, p=1, groups=groups)

        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                WSConv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.GroupNorm(min(groups, out_ch) if out_ch % min(groups, out_ch) == 0 else 1, out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.conv2(self.conv1(x))
        if self.proj is not None:
            identity = self.proj(identity)
        return F.silu(out + identity)

class Encoder(nn.Module):
    def __init__(self, in_ch=3, base=64, groups=32):
        super().__init__()
        self.stem = nn.Sequential(
            WSConv2d(in_ch, base, kernel_size=7, stride=2, padding=3, bias=False),
            # <-- WS: İlk büyük conv (7x7) WS yapıldı (başlangıç ölçek stabil)

            nn.GroupNorm(min(groups, base) if base % min(groups, base) == 0 else 1, base),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)  # /4

        self.stage1 = nn.Sequential(
            WSResBlock(base, base, stride=1, groups=groups),
            WSResBlock(base, base, stride=1, groups=groups),
        )
        self.stage2 = nn.Sequential(
            WSResBlock(base, base*2, stride=2, groups=groups),  # /8
            WSResBlock(base*2, base*2, stride=1, groups=groups),
        )
        self.stage3 = nn.Sequential(
            WSResBlock(base*2, base*4, stride=2, groups=groups),  # /16
            WSResBlock(base*4, base*4, stride=1, groups=groups),
        )
        self.stage4 = nn.Sequential(
            WSResBlock(base*4, base*8, stride=2, groups=groups),  # /32
            WSResBlock(base*8, base*8, stride=1, groups=groups),
        )

    def forward(self, x):
        s0 = self.stem(x)       # /2
        s1 = self.pool(s0)      # /4
        s1 = self.stage1(s1)    # /4
        s2 = self.stage2(s1)    # /8
        s3 = self.stage3(s2)    # /16
        s4 = self.stage4(s3)    # /32
        return s0, s1, s2, s3, s4

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, groups=32):
        super().__init__()
        self.reduce = WSGNAct(in_ch, out_ch, k=1, s=1, p=0, groups=groups)
        self.refine = nn.Sequential(
            WSGNAct(out_ch + skip_ch, out_ch, k=3, s=1, p=1, groups=groups),
            WSGNAct(out_ch, out_ch, k=3, s=1, p=1, groups=groups),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.reduce(x)
        if x.shape[-2:] != skip.shape[-2:]:
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.refine(x)

class WSResUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=1, base=64, groups=32):
        super().__init__()
        self.enc = Encoder(in_ch=in_ch, base=base, groups=groups)
        self.bridge = nn.Sequential(
            WSGNAct(base*8, base*8, k=3, s=1, p=1, groups=groups),
            WSGNAct(base*8, base*8, k=3, s=1, p=1, groups=groups),
        )

        self.up4 = UpBlock(base*8, base*4, base*4, groups=groups)  # /32 -> /16
        self.up3 = UpBlock(base*4, base*2, base*2, groups=groups)  # /16 -> /8
        self.up2 = UpBlock(base*2, base,   base,   groups=groups)  # /8  -> /4
        self.up1 = UpBlock(base,   base,   base//2, groups=groups) # /4  -> /2

        self.head = nn.Conv2d(base//2, num_classes, kernel_size=1)

    def forward(self, x):
        s0, s1, s2, s3, s4 = self.enc(x)
        x = self.bridge(s4)
        x = self.up4(x, s3)
        x = self.up3(x, s2)
        x = self.up2(x, s1)
        x = self.up1(x, s0)
        return self.head(x)
    
if __name__ == "__main__":
    model = WSResUNet(in_ch=3, num_classes=2, base=32, groups=32)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("Output:", y.shape)  