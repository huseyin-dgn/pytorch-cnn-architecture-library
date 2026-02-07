# strong_segmentation.py
# GN + WS + CBAM + (opsiyonel) Dilated DWConv + Residual
# Encoder-Decoder (UNet-lite) + Skip connections
# Pure PyTorch. Output: (B, num_classes, H, W)

import torch
import torch.nn as nn
import torch.nn.functional as F

def pick_gn_groups(ch: int, max_groups: int = 32) -> int:
    g = min(max_groups, ch)
    while g > 1 and (ch % g) != 0:
        g -= 1
    return g

class WSConv2d(nn.Conv2d):
    def __init__(self, *args, ws_eps: float = 1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws_eps = ws_eps

    def forward(self, x):
        w = self.weight
        w_mean = w.mean(dim=(1, 2, 3), keepdim=True)
        w = w - w_mean
        w_var = w.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        w = w / torch.sqrt(w_var + self.ws_eps)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class GNWS(nn.Module):
    def __init__(self, ch: int, max_gn_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        g = pick_gn_groups(ch, max_groups=max_gn_groups)
        self.gn = nn.GroupNorm(g, ch, eps=eps, affine=True)

    def forward(self, x):
        return self.gn(x)

class WSConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1, dilation=1, act=True, max_gn_groups=32):
        super().__init__()
        self.conv = WSConv2d(
            in_ch, out_ch,
            kernel_size=k, stride=s, padding=p,
            dilation=dilation, groups=groups, bias=False
        )
        self.norm = GNWS(out_ch, max_gn_groups=max_gn_groups)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        a = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * a

class SpatialAttention(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(2, 1, k, padding=p, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        a = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * a

class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, sa_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(sa_kernel)

    def forward(self, x):
        return self.sa(self.ca(x))

class SegStrongBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expand: float = 2.0,
        dilation: int = 1,
        max_gn_groups: int = 32
    ):
        super().__init__()
        mid = int(round(in_ch * expand))
        self.use_res = (stride == 1 and in_ch == out_ch)

        self.pw1 = WSConvGNAct(in_ch, mid, k=1, s=1, p=0, act=True, max_gn_groups=max_gn_groups)

        self.dw = WSConvGNAct(
            mid, mid,
            k=3, s=stride, p=dilation,
            groups=mid, dilation=dilation,
            act=True, max_gn_groups=max_gn_groups
        )

        self.cbam = CBAM(mid, reduction=16, sa_kernel=7)

        self.pw2 = WSConvGNAct(mid, out_ch, k=1, s=1, p=0, act=False, max_gn_groups=max_gn_groups)

    def forward(self, x):
        identity = x
        x = self.pw1(x)
        x = self.dw(x)
        x = self.cbam(x)
        x = self.pw2(x)
        if self.use_res:
            x = x + identity
        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, repeats=2, dilation=1, max_gn_groups=32):
        super().__init__()
        blocks = [SegStrongBlock(in_ch, out_ch, stride=2, expand=2.0, dilation=dilation, max_gn_groups=max_gn_groups)]
        for _ in range(repeats - 1):
            blocks.append(SegStrongBlock(out_ch, out_ch, stride=1, expand=2.0, dilation=dilation, max_gn_groups=max_gn_groups))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, repeats=2, max_gn_groups=32):
        super().__init__()
        self.fuse1 = WSConvGNAct(in_ch + skip_ch, out_ch, k=1, s=1, p=0, act=True, max_gn_groups=max_gn_groups)

        blocks = [SegStrongBlock(out_ch, out_ch, stride=1, expand=2.0, dilation=1, max_gn_groups=max_gn_groups)]
        for _ in range(repeats - 1):
            blocks.append(SegStrongBlock(out_ch, out_ch, stride=1, expand=2.0, dilation=1, max_gn_groups=max_gn_groups))
        self.refine = nn.Sequential(*blocks)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.fuse1(x)
        x = self.refine(x)
        return x

class StrongSegmentationModel(nn.Module):
    def __init__(self, num_classes: int, width: int = 32, max_gn_groups: int = 32):
        super().__init__()

        self.stem = nn.Sequential(
            WSConvGNAct(3, width, k=3, s=1, p=1, act=True, max_gn_groups=max_gn_groups),
            WSConvGNAct(width, width, k=3, s=1, p=1, act=True, max_gn_groups=max_gn_groups),
        )

        self.d1 = Down(width,   width * 2, repeats=2, dilation=1, max_gn_groups=max_gn_groups)  # /2
        self.d2 = Down(width*2, width * 4, repeats=2, dilation=1, max_gn_groups=max_gn_groups)  # /4

        self.d3 = Down(width*4, width * 8, repeats=2, dilation=2, max_gn_groups=max_gn_groups)  # /8

        self.bottleneck = nn.Sequential(
            SegStrongBlock(width*8, width*8, stride=1, expand=2.0, dilation=2, max_gn_groups=max_gn_groups),
            SegStrongBlock(width*8, width*8, stride=1, expand=2.0, dilation=2, max_gn_groups=max_gn_groups),
        )
        self.u2 = Up(in_ch=width*8, skip_ch=width*4, out_ch=width*4, repeats=2, max_gn_groups=max_gn_groups)  # /4
        self.u1 = Up(in_ch=width*4, skip_ch=width*2, out_ch=width*2, repeats=2, max_gn_groups=max_gn_groups)  # /2
        self.u0 = Up(in_ch=width*2, skip_ch=width,   out_ch=width,   repeats=2, max_gn_groups=max_gn_groups)  # /1

        self.head = nn.Conv2d(width, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.stem(x)   # (B,width,H,W)
        x1 = self.d1(x0)    # (B,2w,H/2,W/2)
        x2 = self.d2(x1)    # (B,4w,H/4,W/4)
        x3 = self.d3(x2)    # (B,8w,H/8,W/8)

        xb = self.bottleneck(x3)

        y2 = self.u2(xb, x2)   # (B,4w,H/4,W/4)
        y1 = self.u1(y2, x1)   # (B,2w,H/2,W/2)
        y0 = self.u0(y1, x0)   # (B,w,H,W)

        return self.head(y0)

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    m = StrongSegmentationModel(num_classes=19, width=32, max_gn_groups=32).eval()
    x = torch.randn(2, 3, 256, 256)
    y = m(x)
    print("logits:", tuple(y.shape))  