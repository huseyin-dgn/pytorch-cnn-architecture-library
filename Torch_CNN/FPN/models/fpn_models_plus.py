import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import DeformConv2d
    _HAS_DEFORM = True
except Exception:
    DeformConv2d = None
    _HAS_DEFORM = False


class DropBlock2D(nn.Module):
    def __init__(self, drop_prob=0.0, block_size=7):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.block_size = int(block_size)

    def forward(self, x):
        if (not self.training) or self.drop_prob <= 0.0:
            return x
        b, c, h, w = x.shape
        if self.block_size > min(h, w):
            return x
        gamma = self.drop_prob * (h * w) / (self.block_size ** 2) / (
            (h - self.block_size + 1) * (w - self.block_size + 1)
        )
        gamma = float(max(0.0, min(1.0, gamma)))
        mask = (torch.rand((b, 1, h, w), device=x.device, dtype=x.dtype) < gamma).to(x.dtype)
        block_mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        block_mask = 1.0 - block_mask
        denom = block_mask.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return x * block_mask / denom


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, bn=True, act="silu", groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p, groups=groups, bias=not bn)
        self.bn = nn.BatchNorm2d(cout) if bn else nn.Identity()
        if act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act is None:
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unknown act={act}")

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DeformConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, bn=True, act="silu"):
        super().__init__()
        self.use_deform = _HAS_DEFORM
        if self.use_deform:
            self.offset = nn.Conv2d(cin, 2 * k * k, kernel_size=k, stride=s, padding=p, bias=True)
            self.conv = DeformConv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=not bn)
        else:
            self.offset = None
            self.conv = nn.Conv2d(cin, cout, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(cout) if bn else nn.Identity()
        if act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act is None:
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unknown act={act}")

    def forward(self, x):
        if self.use_deform:
            off = self.offset(x)
            x = self.conv(x, off)
        else:
            x = self.conv(x)
        return self.act(self.bn(x))


class SE(nn.Module):
    def __init__(self, channels, reduction=16, min_hidden=4):
        super().__init__()
        hidden = max(min_hidden, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_k=7, min_hidden=4):
        super().__init__()
        hidden = max(min_hidden, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.chan_gate = nn.Sigmoid()
        self.spatial = nn.Conv2d(2, 1, spatial_k, padding=spatial_k // 2, bias=True)
        self.spa_gate = nn.Sigmoid()

    def forward(self, x):
        ca = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        x = x * self.chan_gate(ca)
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        sa = torch.cat([avg, mx], dim=1)
        x = x * self.spa_gate(self.spatial(sa))
        return x


class CoordAtt(nn.Module):
    def __init__(self, channels, reduction=32, min_hidden=8):
        super().__init__()
        hidden = max(min_hidden, channels // reduction)
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden, channels, 1, bias=True)
        self.conv_w = nn.Conv2d(hidden, channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.transpose(2, 3)
        a_h = self.gate(self.conv_h(y_h))
        a_w = self.gate(self.conv_w(y_w))
        return x * a_h * a_w


def build_attention(attn: str, channels: int):
    a = (attn or "none").lower()
    if a == "none":
        return nn.Identity()
    if a == "se":
        return SE(channels)
    if a == "cbam":
        return CBAM(channels)
    if a == "coord":
        return CoordAtt(channels)
    raise ValueError("attn must be one of: none | se | cbam | coord")


class ResidualBlock(nn.Module):
    def __init__(
        self,
        cin,
        cout,
        stride=1,
        bn=True,
        act="silu",
        conv2_type="standard",
        attn="none",
        dropblock_p=0.0,
        dropblock_size=7,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(cin, cout, k=3, s=stride, p=1, bn=bn, act=act)

        if conv2_type == "deform":
            self.conv2 = DeformConvBNAct(cout, cout, k=3, s=1, p=1, bn=bn, act=None)
        else:
            self.conv2 = ConvBNAct(cout, cout, k=3, s=1, p=1, bn=bn, act=None)

        self.attn = build_attention(attn, cout)
        self.dropblock = DropBlock2D(drop_prob=dropblock_p, block_size=dropblock_size)

        if stride != 1 or cin != cout:
            self.skip = ConvBNAct(cin, cout, k=1, s=stride, p=0, bn=bn, act=None)
        else:
            self.skip = nn.Identity()

        if act == "silu":
            self.out_act = nn.SiLU(inplace=True)
        elif act == "relu":
            self.out_act = nn.ReLU(inplace=True)
        else:
            self.out_act = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv2(self.conv1(x))
        out = self.attn(out)
        out = self.dropblock(out)
        out = out + identity
        return self.out_act(out)


class Backbone_C2C5(nn.Module):
    def __init__(
        self,
        in_ch=3,
        base_ch=64,
        layers=(2, 2, 2, 2),
        bn=True,
        act="silu",
        attn_type="cbam",
        dropblock_p=0.10,
        dropblock_size=7,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, base_ch, k=3, s=2, p=1, bn=bn, act=act),
            ConvBNAct(base_ch, base_ch, k=3, s=1, p=1, bn=bn, act=act),
        )

        ch2 = base_ch * 2
        ch3 = base_ch * 4
        ch4 = base_ch * 8
        ch5 = base_ch * 16
        self.out_channels_list = [ch2, ch3, ch4, ch5]

        self.stage2 = self._make_stage(
            base_ch, ch2, n=layers[0], first_stride=2, bn=bn, act=act,
            conv2_type="standard", attn="none", dropblock_p=0.0, dropblock_size=dropblock_size
        )

        self.stage3 = self._make_stage(
            ch2, ch3, n=layers[1], first_stride=2, bn=bn, act=act,
            conv2_type="standard", attn=attn_type, dropblock_p=dropblock_p, dropblock_size=dropblock_size
        )

        self.stage4 = self._make_stage(
            ch3, ch4, n=layers[2], first_stride=2, bn=bn, act=act,
            conv2_type="deform", attn=attn_type, dropblock_p=dropblock_p, dropblock_size=dropblock_size
        )

        self.stage5 = self._make_stage(
            ch4, ch5, n=layers[3], first_stride=2, bn=bn, act=act,
            conv2_type="standard", attn="none", dropblock_p=0.0, dropblock_size=dropblock_size
        )

    def _make_stage(self, cin, cout, n, first_stride, bn, act, conv2_type, attn, dropblock_p, dropblock_size):
        blocks = [
            ResidualBlock(
                cin, cout, stride=first_stride, bn=bn, act=act,
                conv2_type=conv2_type, attn=attn,
                dropblock_p=dropblock_p, dropblock_size=dropblock_size
            )
        ]
        for _ in range(n - 1):
            blocks.append(
                ResidualBlock(
                    cout, cout, stride=1, bn=bn, act=act,
                    conv2_type=conv2_type, attn=attn,
                    dropblock_p=dropblock_p, dropblock_size=dropblock_size
                )
            )
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        C2 = self.stage2(x)
        C3 = self.stage3(C2)
        C4 = self.stage4(C3)
        C5 = self.stage5(C4)
        return [C2, C3, C4, C5]


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) — klasik top-down yapı.

    Girdi:
      feats = [C2, C3, C4, C5]
      - C2: en yüksek çözünürlük (stride~4)  -> küçük objeler için detay
      - C3: stride~8
      - C4: stride~16
      - C5: en düşük çözünürlük (stride~32) -> daha semantik / büyük objeler

    Çıktı:
      [P2, P3, P4, P5]
      - Hepsinin kanal sayısı out_channels'tır (ör. 256)
      - Top-down: P5 -> P4 -> P3 -> P2 (upsample + add)
      - Smooth: her P seviyesine 3x3 conv uygulanır
    """
    def __init__(self, in_channels_list, out_channels=256, upsample_mode="nearest", use_bn=False, use_act=False):
        super().__init__()
        self.upsample_mode = upsample_mode
        act = "silu" if use_act else None

        # Lateral 1x1 conv: C2-C5 kanallarını out_channels'a eşitle (channel alignment)
        self.lateral = nn.ModuleList([
            ConvBNAct(cin, out_channels, k=1, s=1, p=0, bn=use_bn, act=act)
            for cin in in_channels_list
        ])

        # Smooth 3x3 conv: fusion sonrası artefact/aliasing azalt, feature'u toparla
        self.smooth = nn.ModuleList([
            ConvBNAct(out_channels, out_channels, k=3, s=1, p=1, bn=use_bn, act=act)
            for _ in in_channels_list
        ])

    def _upsample(self, x, ref):
        # Üst seviyeyi, referans feature'un HxW boyutuna getir (rounding mismatch riskini azaltır)
        if self.upsample_mode == "bilinear":
            return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return F.interpolate(x, size=ref.shape[-2:], mode=self.upsample_mode)

    def forward(self, feats):
        assert len(feats) == 4, "FPN expects [C2,C3,C4,C5]"
        C2, C3, C4, C5 = feats

        # 1) Lateral projeksiyon: C -> L (kanal eşitleme)
        L2 = self.lateral[0](C2)
        L3 = self.lateral[1](C3)
        L4 = self.lateral[2](C4)
        L5 = self.lateral[3](C5)

        # 2) Top-down: üst semantiği alta taşı (upsample + add)
        P5 = L5
        P4 = L4 + self._upsample(P5, L4)
        P3 = L3 + self._upsample(P4, L3)
        P2 = L2 + self._upsample(P3, L2)

        # 3) Smooth: birleşim sonrası temizle (3x3)
        P2 = self.smooth[0](P2)
        P3 = self.smooth[1](P3)
        P4 = self.smooth[2](P4)
        P5 = self.smooth[3](P5)

        return [P2, P3, P4, P5]


class FullModel(nn.Module):
    def __init__(
        self,
        in_ch=3,
        base_ch=64,
        layers=(2, 2, 2, 2),
        bn=True,
        act="silu",
        attn_type="cbam",
        dropblock_p=0.10,
        dropblock_size=7,
        fpn_out=256,
        upsample_mode="nearest",
        fpn_use_bn=False,
        fpn_use_act=False,
    ):
        super().__init__()
        self.backbone = Backbone_C2C5(
            in_ch=in_ch,
            base_ch=base_ch,
            layers=layers,
            bn=bn,
            act=act,
            attn_type=attn_type,
            dropblock_p=dropblock_p,
            dropblock_size=dropblock_size,
        )
        self.neck = FPN(
            in_channels_list=self.backbone.out_channels_list,
            out_channels=fpn_out,
            upsample_mode=upsample_mode,
            use_bn=fpn_use_bn,
            use_act=fpn_use_act,
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.neck(feats)


if __name__ == "__main__":
    model = FullModel(
        in_ch=3,
        base_ch=32,
        layers=(2, 2, 2, 2),
        attn_type="cbam",
        dropblock_p=0.10,
        dropblock_size=5,
        fpn_out=128,
        upsample_mode="nearest",
    )
    x = torch.randn(1, 3, 256, 256)
    P2, P3, P4, P5 = model(x)
    print("DeformConv available:", _HAS_DEFORM)
    print(P2.shape, P3.shape, P4.shape, P5.shape)