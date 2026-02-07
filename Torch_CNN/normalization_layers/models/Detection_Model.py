# # # # # # # # # #  GN + WS + CBAM + Coordinate Attention + Dynamic(CondConv-lite) + Lite PAN-FPN  # # # # # # # # # # 
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
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1, act=True, max_gn_groups=32):
        super().__init__()
        self.conv = WSConv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
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

class CoordAtt(nn.Module):
    def __init__(self, channels: int, reduction: int = 32, max_gn_groups: int = 32):
        super().__init__()
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, 1, bias=False)
        self.gn1 = GNWS(mip, max_gn_groups=max_gn_groups)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x_h = F.adaptive_avg_pool2d(x, (h, 1))                 
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)  

        y = torch.cat([x_h, x_w], dim=2) 
        y = self.act(self.gn1(self.conv1(y)))

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)  

        a_h = torch.sigmoid(self.conv_h(y_h))  
        a_w = torch.sigmoid(self.conv_w(y_w))  

        return x * a_h * a_w

class DynamicConv1x1(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, K: int = 4):
        super().__init__()
        self.K = K
        self.experts = nn.ModuleList([nn.Conv2d(in_ch, out_ch, 1, bias=False) for _ in range(K)])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, K, 1, bias=True),
        )

    def forward(self, x):
        g = self.gate(x).flatten(2).squeeze(-1) 
        g = torch.softmax(g, dim=1)

        out = 0.0
        for k, conv in enumerate(self.experts):
            out = out + conv(x) * g[:, k].view(-1, 1, 1, 1)
        return out

class DetStrongBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand: float = 2.0,
                 max_gn_groups: int = 32, Kdyn: int = 4):
        super().__init__()
        mid = int(round(in_ch * expand))
        self.use_res = (stride == 1 and in_ch == out_ch)

        self.pw_dyn = DynamicConv1x1(in_ch, mid, K=Kdyn)
        self.pw_norm = GNWS(mid, max_gn_groups=max_gn_groups)
        self.pw_act = nn.SiLU(inplace=True)

        self.dw = WSConvGNAct(mid, mid, k=3, s=stride, p=1, groups=mid, act=True, max_gn_groups=max_gn_groups)

        self.cbam = CBAM(mid, reduction=16, sa_kernel=7)
        self.ca = CoordAtt(mid, reduction=32, max_gn_groups=max_gn_groups)
        self.pw_proj = WSConvGNAct(mid, out_ch, k=1, s=1, p=0, groups=1, act=False, max_gn_groups=max_gn_groups)

    def forward(self, x):
        identity = x
        x = self.pw_act(self.pw_norm(self.pw_dyn(x)))
        x = self.dw(x)
        x = self.cbam(x)
        x = self.ca(x)
        x = self.pw_proj(x)
        if self.use_res:
            x = x + identity
        return x

class StrongBackbone(nn.Module):
    def __init__(self, width: int = 48, max_gn_groups: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            WSConvGNAct(3, width, k=3, s=2, p=1, max_gn_groups=max_gn_groups),  # /2
            WSConvGNAct(width, width, k=3, s=1, p=1, max_gn_groups=max_gn_groups),
        )

        self.s2 = nn.Sequential(
            DetStrongBlock(width, width * 2, stride=2, expand=2.0, max_gn_groups=max_gn_groups),   # /4
            DetStrongBlock(width * 2, width * 2, stride=1, expand=2.0, max_gn_groups=max_gn_groups),
        )
        self.s3 = nn.Sequential(
            DetStrongBlock(width * 2, width * 4, stride=2, expand=2.0, max_gn_groups=max_gn_groups),  # /8  (P3)
            DetStrongBlock(width * 4, width * 4, stride=1, expand=2.0, max_gn_groups=max_gn_groups),
        )
        self.s4 = nn.Sequential(
            DetStrongBlock(width * 4, width * 8, stride=2, expand=2.0, max_gn_groups=max_gn_groups),  # /16 (P4)
            DetStrongBlock(width * 8, width * 8, stride=1, expand=2.0, max_gn_groups=max_gn_groups),
        )
        self.s5 = nn.Sequential(
            DetStrongBlock(width * 8, width * 16, stride=2, expand=2.0, max_gn_groups=max_gn_groups), # /32 (P5)
            DetStrongBlock(width * 16, width * 16, stride=1, expand=2.0, max_gn_groups=max_gn_groups),
        )

        self.out_channels = (width * 4, width * 8, width * 16)

    def forward(self, x):
        x = self.stem(x)
        x = self.s2(x)
        p3 = self.s3(x)
        p4 = self.s4(p3)
        p5 = self.s5(p4)
        return p3, p4, p5

class Fuse(nn.Module):
    def __init__(self, in_ch, out_ch, max_gn_groups=32):
        super().__init__()
        self.proj = WSConvGNAct(in_ch, out_ch, k=1, s=1, p=0, max_gn_groups=max_gn_groups)

    def forward(self, x):
        return self.proj(x)

class LitePANFPN(nn.Module):
    def __init__(self, c3, c4, c5, out_ch=160, max_gn_groups=32):
        super().__init__()
        self.l5 = WSConvGNAct(c5, out_ch, k=1, s=1, p=0, max_gn_groups=max_gn_groups)
        self.l4 = WSConvGNAct(c4, out_ch, k=1, s=1, p=0, max_gn_groups=max_gn_groups)
        self.l3 = WSConvGNAct(c3, out_ch, k=1, s=1, p=0, max_gn_groups=max_gn_groups)

        self.fuse54 = Fuse(out_ch + out_ch, out_ch, max_gn_groups=max_gn_groups)
        self.fuse43 = Fuse(out_ch + out_ch, out_ch, max_gn_groups=max_gn_groups)

        self.down34 = WSConvGNAct(out_ch, out_ch, k=3, s=2, p=1, max_gn_groups=max_gn_groups)
        self.fuse34 = Fuse(out_ch + out_ch, out_ch, max_gn_groups=max_gn_groups)

        self.down45 = WSConvGNAct(out_ch, out_ch, k=3, s=2, p=1, max_gn_groups=max_gn_groups)
        self.fuse45 = Fuse(out_ch + out_ch, out_ch, max_gn_groups=max_gn_groups)

    def forward(self, p3, p4, p5):
        p5 = self.l5(p5)
        p4 = self.l4(p4)
        p3 = self.l3(p3)

        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p4_td = self.fuse54(torch.cat([p4, p5_up], dim=1))

        p4_up = F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest")
        p3_td = self.fuse43(torch.cat([p3, p4_up], dim=1))

        p3_dn = self.down34(p3_td)
        p4_out = self.fuse34(torch.cat([p4_td, p3_dn], dim=1))

        p4_dn = self.down45(p4_out)
        p5_out = self.fuse45(torch.cat([p5, p4_dn], dim=1))

        return p3_td, p4_out, p5_out

class DetHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, max_gn_groups=32):
        super().__init__()
        self.t1 = WSConvGNAct(in_ch, in_ch, k=3, s=1, p=1, max_gn_groups=max_gn_groups)
        self.t2 = WSConvGNAct(in_ch, in_ch, k=3, s=1, p=1, max_gn_groups=max_gn_groups)

        self.cls = nn.Conv2d(in_ch, num_classes, 1)
        self.box = nn.Conv2d(in_ch, 4, 1)
        self.obj = nn.Conv2d(in_ch, 1, 1)

    def forward(self, x):
        x = self.t2(self.t1(x))
        return self.cls(x), self.box(x), self.obj(x)

class StrongDetector(nn.Module):
    def __init__(self, num_classes: int = 80, width: int = 48, neck_ch: int = 160, max_gn_groups: int = 32):
        super().__init__()
        self.backbone = StrongBackbone(width=width, max_gn_groups=max_gn_groups)
        c3, c4, c5 = self.backbone.out_channels

        self.neck = LitePANFPN(c3, c4, c5, out_ch=neck_ch, max_gn_groups=max_gn_groups)

        self.h3 = DetHead(neck_ch, num_classes, max_gn_groups=max_gn_groups)
        self.h4 = DetHead(neck_ch, num_classes, max_gn_groups=max_gn_groups)
        self.h5 = DetHead(neck_ch, num_classes, max_gn_groups=max_gn_groups)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        f3, f4, f5 = self.neck(p3, p4, p5)

        cls3, box3, obj3 = self.h3(f3)
        cls4, box4, obj4 = self.h4(f4)
        cls5, box5, obj5 = self.h5(f5)

        return {
            "P3": (cls3, box3, obj3),
            "P4": (cls4, box4, obj4),
            "P5": (cls5, box5, obj5),
        }

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    m = StrongDetector(num_classes=20, width=32, neck_ch=128, max_gn_groups=32).eval()
    x = torch.randn(2, 3, 256, 256)
    out = m(x)
    for k, (c, b, o) in out.items():
        print(k, "cls", tuple(c.shape), "box", tuple(b.shape), "obj", tuple(o.shape))