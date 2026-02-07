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
        t_min: float = 0.5,
        t_max: float = 3.0,
        router_temperature: float = 1.5,
        beta_ca: float = 0.35,
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

        self.use_spatial_gate = bool(use_spatial_gate)
        self.spatial_gate_beta = float(spatial_gate_beta)
        if self.use_spatial_gate:
            self.spatial_gate_dw = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
            self.spatial_gate_pw = nn.Conv2d(in_channels, in_channels, 1, bias=True)

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
        scale = scale.clamp(self.scale_min, self.scale_max)
        out = x * scale

        if self.use_spatial_gate:
            sg = self.spatial_gate_pw(self.spatial_gate_dw(x))
            sg = F.hardsigmoid(sg, inplace=False)
            sg = 1.0 + self.spatial_gate_beta * (sg - 1.0)
            out = out * sg
        return out

class CBAMChannelPlusCoord(nn.Module):
    def __init__(self, channels: int, residual: bool = False, return_maps: bool = False, monitor: bool = False, **kwargs):
        super().__init__()
        self.return_maps = bool(return_maps)
        self.residual = bool(residual)
        self.monitor = bool(monitor)

        self.ca = ChannelAttentionFusionT(channels=channels, return_fusion_weights=self.return_maps, **kwargs)
        self.coord = CoordinateAttPlus(in_channels=channels, **{k: v for k, v in kwargs.items() if k.startswith("coord_")} if False else {})

    def forward(self, x: torch.Tensor):
        y_ca, _ = self.ca(x)
        y = self.coord(y_ca)
        return y

class BasicResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm="gn", attn=None):
        super().__init__()
        self.norm_type = norm.lower()
        N = lambda c: make_norm(self.norm_type, c)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = N(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.norm2 = N(out_ch)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                N(out_ch)
            )

        self.attn = attn

    def forward(self, x):
        skip = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))

        if self.attn is not None:
            tmp = self.attn(out)
            out = tmp[0] if isinstance(tmp, (tuple, list)) else tmp

        if self.downsample is not None:
            skip = self.downsample(skip)

        out = self.relu(out + skip)
        return out

class MiniResNet(nn.Module):
    def __init__(self, num_classes=10, base_ch=64, layers=(2, 2, 2, 2), norm="gn", attn_factory=None):
        super().__init__()
        self.norm = norm.lower()

        self.stem = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, stride=1, padding=1, bias=False),
            make_norm(self.norm, base_ch),
            nn.ReLU(inplace=False),
        )

        ch1, ch2, ch3, ch4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.stage1 = self._make_stage(ch1, ch1, layers[0], stride=1, norm=self.norm, attn_factory=attn_factory, stage_idx=1)
        self.stage2 = self._make_stage(ch1, ch2, layers[1], stride=2, norm=self.norm, attn_factory=attn_factory, stage_idx=2)
        self.stage3 = self._make_stage(ch2, ch3, layers[2], stride=2, norm=self.norm, attn_factory=attn_factory, stage_idx=3)
        self.stage4 = self._make_stage(ch3, ch4, layers[3], stride=2, norm=self.norm, attn_factory=attn_factory, stage_idx=4)

        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch4, num_classes))

    def _make_stage(self, in_ch, out_ch, n_blocks, stride, norm, attn_factory, stage_idx):
        blocks = []
        for i in range(n_blocks):
            s = stride if i == 0 else 1
            ch_in = in_ch if i == 0 else out_ch
            attn = attn_factory(out_ch, stage_idx, i) if attn_factory is not None else None
            blocks.append(BasicResidualBlock(ch_in, out_ch, stride=s, norm=norm, attn=attn))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)

def make_attn_factory_all():
    def attn_factory(ch, stage_idx, block_idx):
        return CBAMChannelPlusCoord(channels=ch, residual=False, return_maps=False, monitor=False)
    return attn_factory

x = torch.randn(2, 3, 32, 32)

model_plain = MiniResNet(num_classes=10, norm="gn", attn_factory=None)
print("plain:", model_plain(x).shape)

model_attn = MiniResNet(num_classes=10, norm="gn", attn_factory=make_attn_factory_all())
print("attn:", model_attn(x).shape)

for name, m in model_attn.named_modules():
    if isinstance(m, BasicResidualBlock):
        print(name, "proj=", m.downsample is not None)