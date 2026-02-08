import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_cnn.blocks.norm import make_norm

def softplus_inverse(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(torch.clamp(torch.exp(y) - 1.0, min=eps))

def _get_gate(gate: str):
    g = gate.lower()
    if g == "sigmoid":
        return torch.sigmoid
    if g == "hardsigmoid":
        return F.hardsigmoid
    raise ValueError("gate: 'sigmoid' | 'hardsigmoid'")

def _get_act(act: str):
    a = act.lower()
    if a == "relu":
        return nn.ReLU(inplace=True)
    if a == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError("act: 'relu' | 'silu'")

class HSwish(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * F.relu6(x + 3.0, inplace=True) / 6.0

class ChannelAttentionFusionT(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        min_hidden: int = 4,
        fusion: str = "softmax",          # "sum" | "softmax"
        gate: str = "sigmoid",            # "sigmoid" | "hardsigmoid"
        temperature: float = 0.9,
        learnable_temperature: bool = False,
        eps: float = 1e-6,
        act: str = "relu",
        bias: bool = True,
        fusion_router_hidden: int = 16,
        return_fusion_weights: bool = False,
        t_min: float = 0.5,
        t_max: float = 3.0,
        router_temperature: float = 1.5,
        beta_ca: float = 0.35,
    ):
        super().__init__()
        if channels < 1:
            raise ValueError("channels >= 1 olmalı.")
        if fusion not in ("sum", "softmax"):
            raise ValueError("fusion: 'sum' | 'softmax'")
        if temperature <= 0:
            raise ValueError("temperature pozitif olmalı.")
        if fusion == "softmax" and fusion_router_hidden < 1:
            raise ValueError("fusion_router_hidden >= 1 olmalı.")
        if t_min <= 0 or t_max <= 0 or t_min > t_max:
            raise ValueError("T clamp aralığı hatalı.")
        if router_temperature <= 0:
            raise ValueError("router_temperature pozitif olmalı.")
        if beta_ca < 0:
            raise ValueError("beta_ca >= 0 olmalı.")

        self.eps = float(eps)
        self.fusion = fusion
        self.return_fusion_weights = bool(return_fusion_weights)
        self.gate_fn = _get_gate(gate)

        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.Tr = float(router_temperature)
        self.beta_ca = float(beta_ca)

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
            t0 = float(temperature)
            lo = self.t_min + self.eps
            hi = self.t_max - self.eps
            if lo >= hi:
                lo = self.t_min
                hi = self.t_max
            t0 = min(max(t0, lo), hi)
            t_inv = softplus_inverse(torch.tensor(t0), eps=self.eps)
            self.t_raw = nn.Parameter(t_inv)
        else:
            self.register_buffer("T", torch.tensor(float(temperature)))

    def get_T(self, x: torch.Tensor) -> torch.Tensor:
        if self.learnable_temperature:
            T = F.softplus(self.t_raw) + self.eps
        else:
            T = self.T
        T = T.to(device=x.device, dtype=x.dtype)
        return T.clamp(self.t_min, self.t_max)

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
            logits = self.fusion_router(s_cat).flatten(1)
            fusion_w = torch.softmax(logits / self.Tr, dim=1)
            w0 = fusion_w[:, 0].view(-1, 1, 1, 1)
            w1 = fusion_w[:, 1].view(-1, 1, 1, 1)
            z = w0 * a + w1 * m

        T = self.get_T(x)
        ca = self.gate_fn(z / T)

        scale_ca = 1.0 + self.beta_ca * (ca - 1.0)
        y = x * scale_ca

        if self.return_fusion_weights:
            return y, ca, fusion_w
        return y, ca

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
        scale_min: float = 0.6,
        scale_max: float = 1.6,
        head_init_std: float = 0.01,
    ):
        super().__init__()

        self.beta = float(beta)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

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
            raise ValueError("act: 'hswish' | 'relu' | 'silu'")

        self.shared_bottleneck_proj = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.shared_bottleneck_norm = make_norm(norm, mid)
        self.shared_bottleneck_refine = nn.Conv2d(mid, mid, 1, bias=False)
        self.shared_bottleneck_refine_norm = make_norm(norm, mid)

        self.h_local_dw = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0),
                                    groups=in_channels, bias=False)
        self.w_local_dw = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1),
                                    groups=in_channels, bias=False)

        d = int(dilation)
        self.h_dilated_dw = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(d, 0),
                                      dilation=(d, 1), groups=in_channels, bias=False)
        self.w_dilated_dw = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, d),
                                      dilation=(1, d), groups=in_channels, bias=False)

        self.h_channel_mixer = nn.Conv2d(in_channels, in_channels, 1, bias=True)
        self.w_channel_mixer = nn.Conv2d(in_channels, in_channels, 1, bias=True)

        self.h_attention_head = nn.Conv2d(mid, in_channels, 1, bias=True)
        self.w_attention_head = nn.Conv2d(mid, in_channels, 1, bias=True)
        nn.init.normal_(self.h_attention_head.weight, mean=0.0, std=float(head_init_std))
        nn.init.normal_(self.w_attention_head.weight, mean=0.0, std=float(head_init_std))
        nn.init.zeros_(self.h_attention_head.bias)
        nn.init.zeros_(self.w_attention_head.bias)

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

        alpha_h = torch.sigmoid(self.alpha_h_raw).to(device=x.device, dtype=x.dtype)
        alpha_w = torch.sigmoid(self.alpha_w_raw).to(device=x.device, dtype=x.dtype)

        scale_h = (1.0 - alpha_h) + alpha_h * attn_h
        scale_w = (1.0 - alpha_w) + alpha_w * attn_w
        scale = scale_h * scale_w

        scale = 1.0 + self.beta * (scale - 1.0)
        scale = scale.clamp(self.scale_min, self.scale_max)
        out = x * scale

        if self.use_spatial_gate:
            sg = self.spatial_gate_pw(self.spatial_gate_dw(x))
            sg = F.hardsigmoid(sg, inplace=False)
            sg = 1.0 + self.spatial_gate_beta * (sg - 1.0)
            out = out * sg

        return out

class CBAMChannelPlusCoord(nn.Module):
    def __init__(self, channels: int, **kwargs):
        super().__init__()
        self.return_maps = bool(kwargs.pop("return_maps", False))

        self.ca = ChannelAttentionFusionT(
            channels=channels,
            reduction=kwargs.pop("ca_reduction", 16),
            min_hidden=kwargs.pop("ca_min_hidden", 4),
            fusion=kwargs.pop("ca_fusion", "softmax"),
            gate=kwargs.pop("ca_gate", "sigmoid"),
            temperature=kwargs.pop("ca_temperature", 0.9),
            learnable_temperature=kwargs.pop("learnable_temperature", False),
            eps=1e-6,
            act=kwargs.pop("ca_act", "relu"),
            bias=True,
            fusion_router_hidden=kwargs.pop("ca_fusion_router_hidden", 16),
            return_fusion_weights=self.return_maps,
            t_min=kwargs.pop("ca_t_min", 0.5),
            t_max=kwargs.pop("ca_t_max", 3.0),
            router_temperature=kwargs.pop("ca_router_temperature", 1.5),
            beta_ca=kwargs.pop("beta_ca", 0.35),
        )
        self.coord = CoordinateAttPlus(
            in_channels=channels,
            reduction=kwargs.pop("coord_reduction", 32),
            min_mid_channels=kwargs.pop("coord_min_mid", 8),
            act=kwargs.pop("coord_act", "hswish"),
            init_alpha=kwargs.pop("coord_init_alpha", 0.7),
            learnable_alpha=kwargs.pop("coord_learnable_alpha", True),
            beta=kwargs.pop("coord_beta", 0.35),
            dilation=kwargs.pop("coord_dilation", 2),
            norm=kwargs.pop("coord_norm", "gn"),
            use_spatial_gate=kwargs.pop("coord_use_spatial_gate", False),
            spatial_gate_beta=kwargs.pop("coord_spatial_gate_beta", 0.35),
            scale_min=kwargs.pop("coord_scale_min", 0.6),
            scale_max=kwargs.pop("coord_scale_max", 1.6),
            head_init_std=kwargs.pop("coord_head_init_std", 0.01),
        )

        self.residual = bool(kwargs.pop("residual", True))
        self.monitor = bool(kwargs.pop("monitor", False))
        self.kurtarma_modu = str(kwargs.pop("kurtarma_modu", "ratio_floor"))
        self.r_min = float(kwargs.pop("r_min", 0.45))
        self.ema_m = float(kwargs.pop("ema_momentum", 0.95))
        self.min_kurtarma_orani = float(kwargs.pop("min_kurtarma_orani", 0.2))
        self.alpha_etkin_min = float(kwargs.pop("alpha_etkin_min", 0.2))

        if kwargs:
            raise ValueError(f"Unknown kwargs: {list(kwargs.keys())}")

        if self.residual:
            eps = 1e-6
            a0 = 0.75
            raw0 = torch.logit(torch.tensor(a0), eps=eps)
            self.register_buffer("alpha_ogrenilen_raw", raw0)
            self.register_buffer("r_ema", torch.tensor(1.0))

    def _extract_tensor(self, out):
        return out[0] if isinstance(out, (tuple, list)) else out

    def forward(self, x: torch.Tensor):
        if self.return_maps:
            y_ca, ca_map, fusion_w = self.ca(x)
        else:
            y_ca, _ = self.ca(x)
            ca_map, fusion_w = None, None

        y = self.coord(y_ca)
        if not self.residual:
            if self.return_maps:
                return y, {"ca_map": ca_map, "fusion_w": fusion_w}
            return y
        out = x + (y - x)

        if self.return_maps:
            return out, {"ca_map": ca_map, "fusion_w": fusion_w}
        return out