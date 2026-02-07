import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus_inverse(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(torch.clamp(torch.exp(y) - 1.0, min=eps))


def _get_gate(gate: str):
    g = gate.lower()
    if g == "sigmoid":
        return torch.sigmoid
    if g == "hardsigmoid":
        return F.hardsigmoid
    raise ValueError("gate 'sigmoid' veya 'hardsigmoid' olmalı.")


def _get_act(act: str):
    a = act.lower()
    if a == "relu":
        return nn.ReLU(inplace=True)
    if a == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError("act 'relu' veya 'silu' olmalı.")


class ChannelAttentionFusionT(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        min_hidden: int = 4,
        fusion: str = "softmax",
        gate: str = "sigmoid",
        temperature: float = 0.9,
        learnable_temperature: bool = False,
        eps: float = 1e-6,
        act: str = "relu",
        bias: bool = True,
        fusion_router_hidden: int = 16,
        return_fusion_weights: bool = False,
    ):
        super().__init__()

        if channels < 1:
            raise ValueError("channels >= 1 olmalı.")
        if reduction < 1:
            raise ValueError("reduction >= 1 olmalı.")
        if fusion not in ("sum", "softmax"):
            raise ValueError("fusion 'sum' veya 'softmax' olmalı.")
        if temperature <= 0:
            raise ValueError("temperature pozitif olmalı.")
        if fusion == "softmax" and fusion_router_hidden < 1:
            raise ValueError("fusion_router_hidden >= 1 olmalı.")

        self.eps = float(eps)
        self.fusion = fusion
        self.return_fusion_weights = bool(return_fusion_weights)
        self.gate_fn = _get_gate(gate)

        hidden = max(int(min_hidden), int(channels) // int(reduction))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=bias)
        self.act = _get_act(act)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=bias)

        if self.fusion == "softmax":
            self.fusion_router = nn.Sequential(
                nn.Conv2d(2 * channels, fusion_router_hidden, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(fusion_router_hidden, 2, kernel_size=1, bias=True),
            )
            last = self.fusion_router[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        else:
            self.fusion_router = None

        self.learnable_temperature = bool(learnable_temperature)
        if self.learnable_temperature:
            t0 = torch.tensor(float(temperature))
            t_inv = softplus_inverse(t0, eps=self.eps)
            self.t_raw = nn.Parameter(t_inv)
        else:
            self.register_buffer("T", torch.tensor(float(temperature)))

    def get_T(self) -> torch.Tensor:
        if self.learnable_temperature:
            return F.softplus(self.t_raw) + self.eps
        return self.T

    def mlp(self, s: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(s)))

    def forward(self, x: torch.Tensor):
        avg_s = self.avg_pool(x)
        max_s = self.max_pool(x)

        a = self.mlp(avg_s)
        m = self.mlp(max_s)

        fusion_w = None
        if self.fusion == "sum":
            z = a + m
        else:
            s_cat = torch.cat([avg_s, max_s], dim=1)
            logits = self.fusion_router(s_cat).flatten(1)  # (B, 2)
            fusion_w = torch.softmax(logits, dim=1)
            w0 = fusion_w[:, 0].view(-1, 1, 1, 1) # view(-1,1,1,1) = “B boyutunu koru, 3 tane 1 ekle”
            w1 = fusion_w[:, 1].view(-1, 1, 1, 1)
            z = w0 * a + w1 * m

        T = self.get_T().to(device=x.device, dtype=x.dtype)
        ca = self.gate_fn(z / T)
        y = x * ca

        if self.return_fusion_weights and (fusion_w is not None):
            return y, ca, fusion_w
        return y, ca


class HSwish(nn.Module):
    def forward(self, x: torch.Tensor):
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

        mid_floor = max(8, min(32, int(in_channels) // 4))
        mid = max(int(min_mid_channels), int(in_channels) // int(reduction))
        mid = max(mid, int(mid_floor))

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

        self.h_local_dw = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), groups=in_channels, bias=False)
        self.w_local_dw = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), groups=in_channels, bias=False)

        d = int(dilation)
        self.h_dilated_dw = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(d, 0), dilation=(d, 1), groups=in_channels, bias=False)
        self.w_dilated_dw = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, d), dilation=(1, d), groups=in_channels, bias=False)

        self.h_channel_mixer = nn.Conv2d(in_channels, in_channels, 1, bias=True)
        self.w_channel_mixer = nn.Conv2d(in_channels, in_channels, 1, bias=True)

        self.h_attention_head = nn.Conv2d(mid, in_channels, 1, bias=True)
        self.w_attention_head = nn.Conv2d(mid, in_channels, 1, bias=True)

        self.beta = float(beta)

        eps = 1e-6
        a0 = float(init_alpha)
        a0 = min(max(a0, eps), 1.0 - eps)
        raw0 = torch.logit(torch.tensor(a0), eps=eps)

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
        _, _, H, W = x.shape

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

        alpha_h = torch.sigmoid(self.alpha_h_raw).to(device=x.device, dtype=x.dtype)
        alpha_w = torch.sigmoid(self.alpha_w_raw).to(device=x.device, dtype=x.dtype)

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


class CBAMChannelPlusCoord(nn.Module):
    def __init__(
        self,
        channels: int,
        ca_reduction: int = 16,
        ca_min_hidden: int = 4,
        ca_fusion: str = "softmax",
        ca_gate: str = "sigmoid",
        ca_temperature: float = 0.9,
        ca_act: str = "relu",
        ca_fusion_router_hidden: int = 16,
        learnable_temperature: bool = False,
        coord_reduction: int = 32,
        coord_min_mid: int = 8,
        coord_act: str = "hswish",
        coord_init_alpha: float = 0.7,
        coord_learnable_alpha: bool = True,
        coord_beta: float = 0.35,
        coord_dilation: int = 2,
        coord_norm: str = "gn",
        coord_use_spatial_gate: bool = False,
        coord_spatial_gate_beta: float = 0.35,
        residual: bool = True,
        alpha_init: float = 0.75,
        learnable_alpha: bool = False,
        return_maps: bool = False,
    ):
        super().__init__()
        if channels < 1:
            raise ValueError("channels >= 1 olmalı.")

        self.return_maps = bool(return_maps)
        self.residual = bool(residual)

        self.ca = ChannelAttentionFusionT(
            channels=channels,
            reduction=ca_reduction,
            min_hidden=ca_min_hidden,
            fusion=ca_fusion,
            gate=ca_gate,
            temperature=ca_temperature,
            learnable_temperature=learnable_temperature,
            eps=1e-6,
            act=ca_act,
            bias=True,
            fusion_router_hidden=ca_fusion_router_hidden,
            return_fusion_weights=self.return_maps,
        )

        self.coord = CoordinateAttPlus(
            in_channels=channels,
            reduction=coord_reduction,
            min_mid_channels=coord_min_mid,
            act=coord_act,
            init_alpha=coord_init_alpha,
            learnable_alpha=coord_learnable_alpha,
            beta=coord_beta,
            dilation=coord_dilation,
            norm=coord_norm,
            use_spatial_gate=coord_use_spatial_gate,
            spatial_gate_beta=coord_spatial_gate_beta,
        )

        if self.residual:
            eps = 1e-6
            a0 = float(alpha_init)
            a0 = min(max(a0, eps), 1.0 - eps)
            raw0 = torch.logit(torch.tensor(a0), eps=eps)
            if learnable_alpha:
                self.alpha_raw = nn.Parameter(raw0)
            else:
                self.register_buffer("alpha_raw", raw0)

    def _alpha(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "alpha_raw"):
            return x.new_tensor(1.0)
        return torch.sigmoid(self.alpha_raw).to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        if self.return_maps:
            y, ca_map, fusion_w = self.ca(x)
            y = self.coord(y)
            out = x + self._alpha(x) * (y - x) if self.residual else y
            coord_stats = self.coord.last_mask_stats()
            return out, ca_map, fusion_w, coord_stats

        y, _ = self.ca(x)
        y = self.coord(y)
        out = x + self._alpha(x) * (y - x) if self.residual else y
        return out


if __name__ == "__main__":
    x = torch.randn(2, 64, 56, 56)
    m = CBAMChannelPlusCoord(channels=64, return_maps=True, residual=True)
    out, ca_map, fusion_w, coord_stats = m(x)
    print("out:", out.shape)
    print("ca_map:", ca_map.shape)
    print("fusion_w:", fusion_w.shape)
    print("coord_stats:", coord_stats)
