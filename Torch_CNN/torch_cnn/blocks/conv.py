import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_cnn.blocks.norm import make_norm

def _make_act(act: str):
    a = (act or "relu").lower()
    if a == "relu":
        return nn.ReLU(inplace=True)
    if a == "silu":
        return nn.SiLU(inplace=True)
    if a in ("none", "identity"):
        return nn.Identity()
    raise ValueError("act: 'relu' | 'silu' | 'none'")

class RoutingMLP(nn.Module):
    def __init__(self, cin: int, K: int = 4, reduction: int = 4, temperature: float = 1.0):
        super().__init__()
        cin = int(cin)
        hidden = max(1, cin // int(reduction))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(cin, hidden)
        self.fc2 = nn.Linear(hidden, int(K))
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.gap(x).flatten(1)          # (B,C)
        h = F.relu(self.fc1(v), inplace=True)
        logits = self.fc2(h)                # (B,K)
        a = F.softmax(logits / self.temperature, dim=1)
        return a                             # (B,K)

class DynamicConv2d(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        k: int = 3,
        stride: int = 1,
        padding: int | None = None,
        K: int = 4,
        reduction: int = 4,
        temperature: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        cin = int(cin); cout = int(cout); k = int(k); stride = int(stride)
        if padding is None:
            padding = k // 2
        padding = int(padding)

        self.stride = stride
        self.padding = padding
        self.K = int(K)

        # (K, cout, cin, k, k)
        self.weight_bank = nn.Parameter(torch.randn(self.K, cout, cin, k, k) * 0.02)
        self.bias_bank = nn.Parameter(torch.zeros(self.K, cout)) if bias else None

        self.router = RoutingMLP(cin=cin, K=self.K, reduction=reduction, temperature=temperature)

    def forward(self, x: torch.Tensor, return_routing: bool = False):
        B = x.size(0)
        a = self.router(x)  # (B,K)

        Wdyn = torch.einsum("bk,kocij->bocij", a, self.weight_bank)

        bdyn = None
        if self.bias_bank is not None:
            bdyn = torch.einsum("bk,kc->bc", a, self.bias_bank)  # (B, cout)

        outs = []
        for i in range(B):
            yi = F.conv2d(
                x[i:i+1],
                Wdyn[i],
                bias=None if bdyn is None else bdyn[i],
                stride=self.stride,
                padding=self.padding,
            )
            outs.append(yi)
        y = torch.cat(outs, dim=0)

        return (y, a) if return_routing else y

class ConvNormAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=None, norm="bn", act="relu"):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False)
        self.norm = make_norm(norm, cout)
        self.act = _make_act(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DynamicConvNormAct(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        k: int = 3,
        stride: int = 1,
        padding: int | None = None,
        K: int = 4,
        reduction: int = 4,
        temperature: float = 1.0,
        norm: str = "bn",
        act: str = "relu",
    ):
        super().__init__()
        self.dyn = DynamicConv2d(
            cin=cin, cout=cout, k=k, stride=stride, padding=padding,
            K=K, reduction=reduction, temperature=temperature, bias=False
        )
        self.norm = make_norm(norm, cout)
        self.act = _make_act(act)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        y, a = self.dyn(x, return_routing=True)
        y = self.act(self.norm(y))
        if return_aux:
            return y, {"routing": a}
        return y

def make_conv_block(
    kind: str,
    cin: int,
    cout: int,
    *,
    k: int = 3,
    stride: int = 1,
    padding: int | None = None,
    norm: str = "bn",
    act: str = "relu",
    # dynamic params
    K: int = 4,
    reduction: int = 4,
    temperature: float = 1.0,
):
    knd = (kind or "standard").lower()
    if knd == "standard":
        return ConvNormAct(cin, cout, k=k, s=stride, p=padding, norm=norm, act=act)
    if knd == "dynamic":
        return DynamicConvNormAct(
            cin, cout, k=k, stride=stride, padding=padding,
            K=K, reduction=reduction, temperature=temperature,
            norm=norm, act=act
        )
    raise ValueError("kind: 'standard' | 'dynamic'")
