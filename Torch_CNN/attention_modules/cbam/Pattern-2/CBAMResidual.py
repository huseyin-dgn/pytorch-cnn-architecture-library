import torch
import torch.nn as nn
import torch.nn.functional as F


def _softplus_inverse(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
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


def _make_odd(k: int) -> int:
    k = int(k)
    if k < 1:
        raise ValueError("Kernel size >= 1 olmalı.")
    return k if (k % 2 == 1) else (k + 1)


class ChannelAttentionFusionT(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        min_hidden: int = 4,
        fusion: str = "softmax",        # "sum" | "softmax"
        gate: str = "sigmoid",          # "sigmoid" | "hardsigmoid"
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        eps: float = 1e-6,
        act: str = "relu",
        bias: bool = True,
        fusion_router_hidden: int = 16,
        return_fusion_weights: bool = False,
    ):
        super().__init__()
        if fusion not in ("sum", "softmax"):
            raise ValueError("fusion 'sum' veya 'softmax' olmalı.")
        if temperature <= 0:
            raise ValueError("temperature pozitif olmalı.")
        if fusion_router_hidden < 1:
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
        else:
            self.fusion_router = None

        self.learnable_temperature = bool(learnable_temperature)
        if self.learnable_temperature:
            t0 = torch.tensor(float(temperature))
            t_inv = _softplus_inverse(t0, eps=self.eps)
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
        avg_s = self.avg_pool(x)  # (B,C,1,1)
        max_s = self.max_pool(x)  # (B,C,1,1)

        a = self.mlp(avg_s)       # (B,C,1,1)
        m = self.mlp(max_s)       # (B,C,1,1)

        fusion_w = None
        if self.fusion == "sum":
            z = a + m
        else:
            s_cat = torch.cat([avg_s, max_s], dim=1)          # (B,2C,1,1)
            logits = self.fusion_router(s_cat).flatten(1)     # (B,2)
            fusion_w = torch.softmax(logits, dim=1)           # (B,2)
            z = fusion_w[:, 0].view(-1, 1, 1, 1) * a + fusion_w[:, 1].view(-1, 1, 1, 1) * m

        T = self.get_T()
        ca = self.gate_fn(z / T)  # (B,C,1,1)
        y = x * ca

        if self.return_fusion_weights and (fusion_w is not None):
            return y, ca, fusion_w
        return y, ca


class _DWPointwiseBranch(nn.Module):
    def __init__(self, in_ch: int, k: int, dilation: int = 1):
        super().__init__()
        k = _make_odd(k)
        dilation = int(dilation)
        if dilation < 1:
            raise ValueError("dilation >= 1 olmalı.")
        pad = dilation * (k - 1) // 2

        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=k, padding=pad, dilation=dilation, groups=in_ch, bias=False
        )
        self.pw = nn.Conv2d(in_ch, 1, kernel_size=1, bias=False)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(s))


class DynamicSpatialAttention(nn.Module):
    def __init__(
        self,
        kernels=(3, 7),
        use_dilated: bool = True,
        dilated_kernel: int = 7,
        dilated_d: int = 2,
        gate: str = "sigmoid",
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        eps: float = 1e-6,
        router_hidden: int = 8,
        bias: bool = True,
        return_router_weights: bool = False,
        coord_norm: str = "minus1to1",  # "minus1to1" | "0to1"
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature pozitif olmalı.")
        if router_hidden < 1:
            raise ValueError("router_hidden >= 1 olmalı.")
        if coord_norm not in ("minus1to1", "0to1"):
            raise ValueError("coord_norm 'minus1to1' veya '0to1' olmalı.")

        self.eps = float(eps)
        self.return_router_weights = bool(return_router_weights)
        self.gate_fn = _get_gate(gate)
        self.coord_norm = coord_norm

        in_ch = 4  # [avg_map, max_map, x_coord, y_coord]

        ks = [_make_odd(int(k)) for k in kernels]
        branches = [_DWPointwiseBranch(in_ch=in_ch, k=k, dilation=1) for k in ks]

        if use_dilated:
            dk = _make_odd(int(dilated_kernel))
            dd = int(dilated_d)
            if dd < 1:
                raise ValueError("dilated_d >= 1 olmalı.")
            branches.append(_DWPointwiseBranch(in_ch=in_ch, k=dk, dilation=dd))

        self.branches = nn.ModuleList(branches)
        self.num_branches = len(self.branches)

        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, router_hidden, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(router_hidden, self.num_branches, kernel_size=1, bias=bias),
        )

        self.learnable_temperature = bool(learnable_temperature)
        if self.learnable_temperature:
            t0 = torch.tensor(float(temperature))
            t_inv = _softplus_inverse(t0, eps=self.eps)
            self.t_raw = nn.Parameter(t_inv)
        else:
            self.register_buffer("T", torch.tensor(float(temperature)))

        self._coord_cache = {}

    def get_T(self) -> torch.Tensor:
        if self.learnable_temperature:
            return F.softplus(self.t_raw) + self.eps
        return self.T

    def _coords(self, B: int, H: int, W: int, device, dtype):
        key = (H, W, str(device), str(dtype), self.coord_norm)
        if key in self._coord_cache:
            xg, yg = self._coord_cache[key]
        else:
            if self.coord_norm == "minus1to1":
                xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
                ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
            else:
                xs = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
                ys = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)

            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            xg = xx.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            yg = yy.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            self._coord_cache[key] = (xg, yg)

        return xg.expand(B, -1, -1, -1), yg.expand(B, -1, -1, -1)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        avg_map = torch.mean(x, dim=1, keepdim=True)       # (B,1,H,W)
        max_map, _ = torch.max(x, dim=1, keepdim=True)     # (B,1,H,W)

        x_coord, y_coord = self._coords(B, H, W, x.device, x.dtype)
        s = torch.cat([avg_map, max_map, x_coord, y_coord], dim=1)  # (B,4,H,W)

        logits = self.router(s).flatten(1)    # (B,K)
        rw = torch.softmax(logits, dim=1)     # (B,K)

        z = torch.stack([br(s) for br in self.branches], dim=1)  # (B,K,1,H,W)
        wlogit = (rw[:, :, None, None, None] * z).sum(dim=1)     # (B,1,H,W)

        T = self.get_T()
        sa = self.gate_fn(wlogit / T)         # (B,1,H,W)
        y = x * sa

        if self.return_router_weights:
            return y, sa, rw
        return y, sa


class CBAMResidualDynamicSA(nn.Module):
    """Wrapper: CA -> SA + opsiyonel residual karışım."""

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        min_hidden: int = 4,
        # CA
        ca_fusion: str = "softmax",
        ca_gate: str = "sigmoid",
        ca_temperature: float = 1.0,
        ca_act: str = "relu",
        ca_fusion_router_hidden: int = 16,
        # SA
        sa_gate: str = "sigmoid",
        sa_temperature: float = 1.0,
        sa_kernels=(3, 7),
        sa_use_dilated: bool = True,
        sa_dilated_kernel: int = 7,
        sa_dilated_d: int = 2,
        sa_router_hidden: int = 8,
        sa_coord_norm: str = "minus1to1",
        # shared
        learnable_temperature: bool = False,
        # residual
        residual: bool = True,
        alpha_init: float = 1.0,
        learnable_alpha: bool = False,
        # outputs
        return_maps: bool = False,
    ):
        
#         CA (Channel) grubu :: :: channels, reduction, min_hidden ,  ca_fusion, ca_gate, ca_temperature, ca_act, ca_fusion_router_hidden 
#         SA (Spatial) grubu :: :: sa_kernels, sa_use_dilated, sa_dilated_kernel, sa_dilated_d , sa_gate, sa_temperature, sa_router_hidden, sa_coord_norm
#         Shared (ortak kontrol) :: :: learnable_temperature (CA ve SA’nın temperature’ını öğrenilebilir yapar)
#         Wrapper davranışı :: :: residual, alpha_init, learnable_alpha, return_maps

        super().__init__()
        self.return_maps = bool(return_maps)
        self.residual = bool(residual)

        self.ca = ChannelAttentionFusionT(
            channels=channels,
            reduction=reduction,
            min_hidden=min_hidden,
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

        self.sa = DynamicSpatialAttention(
            kernels=sa_kernels,
            use_dilated=sa_use_dilated,
            dilated_kernel=sa_dilated_kernel,
            dilated_d=sa_dilated_d,
            gate=sa_gate,
            temperature=sa_temperature,
            learnable_temperature=learnable_temperature,
            eps=1e-6,
            router_hidden=sa_router_hidden,
            bias=True,
            return_router_weights=self.return_maps,
            coord_norm=sa_coord_norm,
        )

        if self.residual:
            if learnable_alpha:
                self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
            else:
                self.register_buffer("alpha", torch.tensor(float(alpha_init)))

    def forward(self, x: torch.Tensor):
        if self.return_maps:
            y, ca, fusion_w = self.ca(x)
            y, sa, router_w = self.sa(y)
            out = x + self.alpha * (y - x) if self.residual else y
            return out, ca, sa, fusion_w, router_w

        y, _ = self.ca(x)
        y, _ = self.sa(y)
        out = x + self.alpha * (y - x) if self.residual else y
        return out


if __name__ == "__main__":
    x = torch.randn(2, 64, 56, 56)
    model = CBAMResidualDynamicSA(
        channels=64,
        return_maps=True,
        residual=True,
        learnable_alpha=False,
        learnable_temperature=True,
        sa_kernels=(3, 5, 7),
        sa_use_dilated=True,
    )
    out, ca, sa, fusion_w, router_w = model(x)
    print("out:", out.shape)
    print("ca:", ca.shape)
    print("sa:", sa.shape)
    print("fusion_w:", fusion_w.shape)
    print("router_w:", router_w.shape)