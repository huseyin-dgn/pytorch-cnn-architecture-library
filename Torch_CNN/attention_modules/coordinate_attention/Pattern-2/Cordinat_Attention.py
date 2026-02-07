import torch
import torch.nn as nn
import torch.nn.functional as F


class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=True) / 6.0


def make_norm(norm: str, ch: int):
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm2d(ch)
    if norm == "gn":
        g = min(32, ch)
        while ch % g != 0 and g > 2:
            g //= 2
        if ch % g != 0:
            g = 2 if (ch % 2 == 0) else 1
        return nn.GroupNorm(g, ch)
    if norm == "none":
        return nn.Identity()
    raise ValueError("norm 'none', 'bn', 'gn' dışında olamaz.")


class CoordinateAttPlus(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction: int = 32,
        min_mid_channels: int = 8,
        act: str = "hswish",
        init_alpha: float = 0.7,
        learnable_alpha: bool = True,
        beta: float = 0.35,
        dilation: int = 2,
        norm: str = "gn",
        use_spatial_gate: bool = False,
        spatial_gate_beta: float = 0.35,
    ):
        super().__init__()

        if in_channels < 1:
            raise ValueError("in_channels >= 1 olmalı.")
        if reduction < 1:
            raise ValueError("reduction >= 1 olmalı.")
        if dilation < 1:
            raise ValueError("dilation >= 1 olmalı.")

        mid = max(int(min_mid_channels), int(in_channels) // int(reduction))

        act_l = act.lower()
        if act_l == "hswish":
            self.act = HSwish()
        elif act_l == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act_l == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError("act 'hswish', 'relu', 'silu' olmalı.")

        self.shared_bottleneck_proj = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.shared_bottleneck_norm = make_norm(norm, mid)
        self.shared_bottleneck_refine = nn.Conv2d(mid, mid, 1, bias=False)
        self.shared_bottleneck_refine_norm = make_norm(norm, mid)

        self.h_local_dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), groups=in_channels, bias=False
        )
        self.w_local_dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), groups=in_channels, bias=False
        )

        d = int(dilation)
        self.h_dilated_dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 1),
            padding=(d, 0),
            dilation=(d, 1),
            groups=in_channels,
            bias=False,
        )
        self.w_dilated_dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(1, 3),
            padding=(0, d),
            dilation=(1, d),
            groups=in_channels,
            bias=False,
        )

        self.h_channel_mixer = nn.Conv2d(in_channels, in_channels, 1, bias=True)
        self.w_channel_mixer = nn.Conv2d(in_channels, in_channels, 1, bias=True)

        self.h_attention_head = nn.Conv2d(mid, in_channels, 1, bias=True)
        self.w_attention_head = nn.Conv2d(mid, in_channels, 1, bias=True)

        self.beta = float(beta)

        eps = 1e-6
        a0 = float(init_alpha)
        a0 = min(max(a0, eps), 1.0 - eps)
        raw0 = torch.log(torch.tensor(a0) / (1.0 - torch.tensor(a0)))

        if learnable_alpha:
            self.alpha_h_raw = nn.Parameter(raw0.clone())
            self.alpha_w_raw = nn.Parameter(raw0.clone())
        else:
            self.register_buffer("alpha_h_raw", raw0.clone())
            self.register_buffer("alpha_w_raw", raw0.clone())

        self.use_spatial_gate = bool(use_spatial_gate)
        self.spatial_gate_beta = float(spatial_gate_beta)
        if self.use_spatial_gate:
            self.spatial_gate_dw = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
            self.spatial_gate_pw = nn.Conv2d(in_channels, in_channels, 1, bias=True)

        self._last_ah = None
        self._last_aw = None

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        h_profile = 0.5 * (x.mean(dim=3, keepdim=True) + x.amax(dim=3, keepdim=True))
        w_profile = 0.5 * (x.mean(dim=2, keepdim=True) + x.amax(dim=2, keepdim=True))

        h_ms = self.h_channel_mixer(self.h_local_dw(h_profile) + self.h_dilated_dw(h_profile))
        w_ms = self.w_channel_mixer(self.w_local_dw(w_profile) + self.w_dilated_dw(w_profile))
        w_ms = w_ms.permute(0, 1, 3, 2)

        hw = torch.cat([h_ms, w_ms], dim=2)

        mid = self.act(self.shared_bottleneck_norm(self.shared_bottleneck_proj(hw)))
        mid = self.act(self.shared_bottleneck_refine_norm(self.shared_bottleneck_refine(mid)))

        mid_h, mid_w = torch.split(mid, [H, W], dim=2)
        mid_w = mid_w.permute(0, 1, 3, 2)

        attn_h = F.hardsigmoid(self.h_attention_head(mid_h), inplace=False)
        attn_w = F.hardsigmoid(self.w_attention_head(mid_w), inplace=False)

        self._last_ah = attn_h.detach()
        self._last_aw = attn_w.detach()

        alpha_h = torch.sigmoid(self.alpha_h_raw)
        alpha_w = torch.sigmoid(self.alpha_w_raw)

        scale_h = (1.0 - alpha_h) + alpha_h * attn_h
        scale_w = (1.0 - alpha_w) + alpha_w * attn_w

        scale = scale_h * scale_w
        scale = 1.0 + self.beta * (scale - 1.0)

        out = x * scale

        if self.use_spatial_gate:
            sg = self.spatial_gate_pw(self.spatial_gate_dw(x))
            sg = F.hardsigmoid(sg, inplace=False)
            sg = 1.0 + self.spatial_gate_beta * (sg - 1.0)
            out = out * sg

        return out

    @torch.no_grad()
    def last_mask_stats(self):
        if (self._last_ah is None) or (self._last_aw is None):
            return None
        ah = self._last_ah
        aw = self._last_aw
        return {
            "a_h": {"min": float(ah.min()), "mean": float(ah.mean()), "max": float(ah.max()), "std": float(ah.std())},
            "a_w": {"min": float(aw.min()), "mean": float(aw.mean()), "max": float(aw.max()), "std": float(aw.std())},
        }


if __name__ == "__main__":
    x = torch.randn(2, 64, 56, 56)
    m = CoordinateAttPlus(64, reduction=32, min_mid_channels=8, norm="gn", dilation=2)
    y = m(x)
    print(y.shape)
    print(m.last_mask_stats())
