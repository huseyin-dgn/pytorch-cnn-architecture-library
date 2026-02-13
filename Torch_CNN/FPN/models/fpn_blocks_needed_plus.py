import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 1) ConvBNAct: Conv -> (BN) -> (Act)
# =========================================================
class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=1, s=1, p=0, bn=True, act=None):
        super().__init__()
        # BN varsa bias gereksiz (BN offset öğrenir), BN yoksa bias açık kalsın
        self.conv = nn.Conv2d(cin, cout, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(cout) if bn else nn.Identity()

        if act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# =========================================================
# 2) Attention blocks (SE / CBAM / CoordAtt)
# =========================================================
class SE(nn.Module):
    def __init__(self, channels, reduction=16, min_hidden=4):
        super().__init__()
        hidden = max(min_hidden, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_k=7, min_hidden=4):
        super().__init__()
        hidden = max(min_hidden, channels // reduction)

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True)
        )
        self.chan_gate = nn.Sigmoid()

        # Spatial attention
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

        x_h = x.mean(dim=3, keepdim=True)              # (B,C,H,1)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)  # (B,C,W,1)

        y = torch.cat([x_h, x_w], dim=2)              # (B,C,H+W,1)
        y = self.act(self.bn1(self.conv1(y)))

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.transpose(2, 3)                     # (B,hidden,1,W)

        a_h = self.gate(self.conv_h(y_h))             # (B,C,H,1)
        a_w = self.gate(self.conv_w(y_w))             # (B,C,1,W)

        return x * a_h * a_w


def build_attention(attn: str, channels: int):
    attn = (attn or "none").lower()
    if attn == "none":
        return nn.Identity()
    if attn == "se":
        return SE(channels)
    if attn == "cbam":
        return CBAM(channels)
    if attn == "coord":
        return CoordAtt(channels)
    raise ValueError("attn must be one of: none | se | cbam | coord")


# =========================================================
# 3) Residual block + attention
# =========================================================
class ResidualAttnBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, attn="none", bn=True, act="silu"):
        super().__init__()

        # Main path
        self.conv1 = ConvBNAct(cin, cout, k=3, s=stride, p=1, bn=bn, act=act)
        self.conv2 = ConvBNAct(cout, cout, k=3, s=1, p=1, bn=bn, act=None)  # add öncesi act yok
        self.attn = build_attention(attn, cout)

        # Skip/projection path
        if stride != 1 or cin != cout:
            self.skip = ConvBNAct(cin, cout, k=1, s=stride, p=0, bn=bn, act=None)
        else:
            self.skip = nn.Identity()

        # Final activation
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
        out = out + identity
        return self.out_act(out)


# =========================================================
# 4) Backbone: returns [C2, C3, C4, C5]
# =========================================================
class ResAttnBackbone(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, layers=(2, 2, 2, 2), attn="cbam", bn=True, act="silu"):
        super().__init__()

        # Stem (H/2)
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, base_ch, k=3, s=2, p=1, bn=bn, act=act),
            ConvBNAct(base_ch, base_ch, k=3, s=1, p=1, bn=bn, act=act),
        )

        ch2 = base_ch * 2
        ch3 = base_ch * 4
        ch4 = base_ch * 8
        ch5 = base_ch * 16

        # Stage2 (H/4) -> C2
        self.stage2 = self._make_stage(base_ch, ch2, n=layers[0], first_stride=2, attn=attn, bn=bn, act=act)
        # Stage3 (H/8) -> C3
        self.stage3 = self._make_stage(ch2, ch3, n=layers[1], first_stride=2, attn=attn, bn=bn, act=act)
        # Stage4 (H/16) -> C4
        self.stage4 = self._make_stage(ch3, ch4, n=layers[2], first_stride=2, attn=attn, bn=bn, act=act)
        # Stage5 (H/32) -> C5
        self.stage5 = self._make_stage(ch4, ch5, n=layers[3], first_stride=2, attn=attn, bn=bn, act=act)

        self.out_channels_list = [ch2, ch3, ch4, ch5]

    def _make_stage(self, cin, cout, n, first_stride, attn, bn, act):
        blocks = [ResidualAttnBlock(cin, cout, stride=first_stride, attn=attn, bn=bn, act=act)]
        for _ in range(n - 1):
            blocks.append(ResidualAttnBlock(cout, cout, stride=1, attn=attn, bn=bn, act=act))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        C2 = self.stage2(x)
        C3 = self.stage3(C2)
        C4 = self.stage4(C3)
        C5 = self.stage5(C4)
        return [C2, C3, C4, C5]


# =========================================================
# 5) FPN 
# =========================================================
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
    def __init__(self, in_channels_list, out_channels=256, upsample_mode="nearest",
                 use_bn=False, use_act=False):
        super().__init__()
        self.upsample_mode = upsample_mode
        act = "silu" if use_act else None

        # Lateral 1x1 conv:
        # Amaç: C2-C5'in kanal boyutlarını out_channels'a eşitlemek (kanal hizalama)
        self.lateral = nn.ModuleList([
            ConvBNAct(cin, out_channels, k=1, s=1, p=0, bn=use_bn, act=act)
            for cin in in_channels_list
        ])

        # Smooth 3x3 conv:
        # Amaç: upsample+add sonrası oluşan artefact/aliasing'i azaltmak, feature'u "temizlemek"
        self.smooth = nn.ModuleList([
            ConvBNAct(out_channels, out_channels, k=3, s=1, p=1, bn=use_bn, act=act)
            for _ in in_channels_list
        ])

    def _upsample(self, x, ref):
        # Üst seviyedeki feature'u, referans feature'un (ref) HxW boyutuna getirir.
        # size=ref.shape[-2:] kullanmak rounding kaynaklı mismatch riskini azaltır.
        if self.upsample_mode == "bilinear":
            return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return F.interpolate(x, size=ref.shape[-2:], mode=self.upsample_mode)

    def forward(self, feats):
        # Backbone'un 4 seviye döndürdüğünü varsayıyoruz.
        assert len(feats) == 4, "FPN expects [C2,C3,C4,C5]"
        C2, C3, C4, C5 = feats

        # 1) Lateral projeksiyon: C2-C5 -> L2-L5 (kanal sayısı eşitlenir)
        L2 = self.lateral[0](C2)
        L3 = self.lateral[1](C3)
        L4 = self.lateral[2](C4)
        L5 = self.lateral[3](C5)

        # 2) Top-down birleştirme:
        # En üstten başlarız (P5=L5) ve her seviyeyi bir alttaki çözünürlüğe upsample edip toplarız.
        P5 = L5
        P4 = L4 + self._upsample(P5, L4)
        P3 = L3 + self._upsample(P4, L3)
        P2 = L2 + self._upsample(P3, L2)

        # 3) Smooth: her P seviyesine 3x3 conv (fusion sonrası temizleme)
        P2 = self.smooth[0](P2)
        P3 = self.smooth[1](P3)
        P4 = self.smooth[2](P4)
        P5 = self.smooth[3](P5)

        return [P2, P3, P4, P5]


# =========================================================
# 6) Full model: Backbone + FPN
# =========================================================
class ResAttnFPN(nn.Module):
    """
    Full model that returns pyramids [P2,P3,P4,P5].
    (Head kısmını sen ekleyeceksin: detect/seg/class head vs.)
    """
    def __init__(
        self,
        in_ch=3,
        base_ch=64,
        layers=(2, 2, 2, 2),
        attn="cbam",              # none | se | cbam | coord
        fpn_out=256,
        upsample_mode="nearest",
        fpn_use_bn=False,
        fpn_use_act=False,
        bn=True,
        act="silu",
    ):
        super().__init__()
        self.backbone = ResAttnBackbone(
            in_ch=in_ch, base_ch=base_ch, layers=layers, attn=attn, bn=bn, act=act
        )
        self.neck = FPN(
            in_channels_list=self.backbone.out_channels_list,
            out_channels=fpn_out,
            upsample_mode=upsample_mode,
            use_bn=fpn_use_bn,
            use_act=fpn_use_act,
        )

    def forward(self, x):
        feats = self.backbone(x)      # [C2,C3,C4,C5]
        pyramids = self.neck(feats)   # [P2,P3,P4,P5]
        return pyramids


# =========================================================
# Quick sanity test
# =========================================================
if __name__ == "__main__":
    model = ResAttnFPN(
        in_ch=3,
        base_ch=32,
        layers=(2, 2, 2, 2),
        attn="cbam",
        fpn_out=128,
        upsample_mode="nearest"
    )

    x = torch.randn(1, 3, 256, 256)
    P2, P3, P4, P5 = model(x)

    print("P2:", P2.shape)  # ~ [1, 128, 64, 64]
    print("P3:", P3.shape)  # ~ [1, 128, 32, 32]
    print("P4:", P4.shape)  # ~ [1, 128, 16, 16]
    print("P5:", P5.shape)  # ~ [1, 128, 8, 8]