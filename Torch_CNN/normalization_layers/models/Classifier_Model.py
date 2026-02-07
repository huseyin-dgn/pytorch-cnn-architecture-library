# # # # # # # # # # # # Classifier Model (GN + WS + CBAM + Residual + Depthwise) # # # # # # # # # # # #

# Bu bölümde:
# - CNN backbone'u **DWConv (depthwise separable)** ile kuracağız
# - Norm: **GroupNorm (GN)**
# - Stabilizasyon: **Weight Standardization (WS)** (Conv ağırlığına)
# - Dikkat: **CBAM**
# - Bağlantı: **Residual**
# - Head: GAP + Linear (classifier)

import torch
import torch.nn as nn
import torch.nn.functional as F

def pick_gn_groups(ch:int,max_groups:int = 32) -> int:
    g = min(max_groups,ch)
    while g > 1 and (ch % g) !=0:
        g -=1
    return g

class WSConv2d(nn.Conv2d):
    def __init__(self, *args, ws_eps: float = 1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws_eps = ws_eps

    def forward(self, x):
        w = self.weight
        w_mean = w.mean(dim=(1,2,3),keepdim=True)
        w = w - w_mean
        w_var = w.pow(2).mean(dim=(1,2,3),keepdim=True)
        w = w / torch.sqrt(w_var + self.ws_eps)
        return F.conv2d(x,w,self.bias,self.stride,self.padding,self.dilation,self.groups)

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * attn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, sa_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=sa_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ## # ## # ## # ## # ## # ##  GN + WS + DWConv + Residual Block # ## # ## # ## # ## # ## # ##  

# Blok tasarımı (MobileNetV2-ish ama GN+WS + CBAM ile):
# 1) PW 1x1 (expand)  -> GN -> SiLU
# 2) DW 3x3 (depthwise) -> GN -> SiLU
# 3) CBAM
# 4) PW 1x1 (project) -> GN
# 5) Residual (stride=1 ve kanal eşitse)

class GNWSConvAct(nn.Module):
    def __init__(self,in_ch,out_ch,k = 1,p = 0,s = 1,groups = 1, act = True ,max_gn_groups=32):
        super().__init__()
        self.conv = WSConv2d(in_ch,out_ch,kernel_size=k,padding=p,stride=s,groups=groups)
        g = pick_gn_groups(out_ch,max_groups=max_gn_groups)
        self.gn = nn.GroupNorm(g,out_ch,eps=1e-5,affine=True)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    
    def forward(self,x):
        return self.act(self.gn(self.conv(x)))
    

class DWResCBAMBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand: float = 2.0, max_gn_groups: int = 32):
        super().__init__()
        mid = int(round(in_ch * expand))
        self.use_res = (stride == 1 and in_ch == out_ch)
        # 1) Expand (1x1)
        self.pw1 = GNWSConvAct(in_ch, mid, k=1, s=1, p=0, groups=1, act=True, max_gn_groups=max_gn_groups)
        # 2) Depthwise (3x3)
        self.dw = GNWSConvAct(mid, mid, k=3, s=stride, p=1, groups=mid, act=True, max_gn_groups=max_gn_groups)
        # 3) CBAM
        self.cbam = CBAM(mid, reduction=16, sa_kernel=7)
        self.pw2 = GNWSConvAct(mid, out_ch, k=1, s=1, p=0, groups=1, act=False, max_gn_groups=max_gn_groups)

    def forward(self, x):
        identity = x
        x = self.pw1(x)
        x = self.dw(x)
        x = self.cbam(x)
        x = self.pw2(x)
        if self.use_res:
            x = x + identity
        return x

class GNWS_CBAM_DWClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 32, max_gn_groups: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            GNWSConvAct(3, width, k=3, s=2, p=1, act=True, max_gn_groups=max_gn_groups),
            GNWSConvAct(width, width, k=3, s=1, p=1, act=True, max_gn_groups=max_gn_groups),
        )
        cfg = [
            (width*2, 2, 2),   # /4
            (width*4, 2, 2),   # /8
            (width*8, 2, 2),   # /16
        ]

        blocks = []
        in_ch = width
        for out_ch, s, r in cfg:
            blocks.append(DWResCBAMBlock(in_ch, out_ch, stride=s, expand=2.0, max_gn_groups=max_gn_groups))
            in_ch = out_ch
            for _ in range(r - 1):
                blocks.append(DWResCBAMBlock(in_ch, out_ch, stride=1, expand=2.0, max_gn_groups=max_gn_groups))
        self.backbone = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    m = GNWS_CBAM_DWClassifier(num_classes=10, width=32).eval()
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print(y.shape)  # (2, 10)