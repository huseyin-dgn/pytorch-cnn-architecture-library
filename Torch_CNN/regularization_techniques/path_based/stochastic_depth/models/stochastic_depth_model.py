import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# 1) Stochastic Depth (DropPath for residual branch)
# =========================
class StochasticDepth(nn.Module):
    """
    Sample-wise Stochastic Depth:
    - mask shape: [B, 1, 1, 1] (broadcast)
    - residual branch'i (F(x)) bazen komple düşürür.
    """
    def __init__(self, p: float = 0.0):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError("p must be in [0,1).")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p == 0.0:
            return x
        q = 1.0 - self.p
        shape = (x.size(0),) + (1,) * (x.ndim - 1)  # [B,1,1,1...]
        mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(q)
        return x * mask / q


# =========================
# 2) Building blocks (Conv-BN-Act, SE, Residual + SD)
# =========================
class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, act="silu"):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (hafif, iyi performans)."""
    def __init__(self, c: int, r: int = 16, min_hidden: int = 8):
        super().__init__()
        hidden = max(min_hidden, c // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, hidden, 1, bias=True)
        self.fc2 = nn.Conv2d(hidden, c, 1, bias=True)

    def forward(self, x):
        w = self.pool(x)
        w = F.silu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class BottleneckSD(nn.Module):
    """
    ResNet Bottleneck + SE + StochasticDepth (iyi, ortalama-üstü blok)
    out = x + SD(F(x))
    """
    def __init__(self, cin: int, cout: int, stride: int = 1, expansion: int = 4,
                 sd_p: float = 0.0, use_se: bool = True, act: str = "silu"):
        super().__init__()
        mid = cout // expansion
        assert mid > 0 and cout % expansion == 0, "cout must be divisible by expansion."

        self.conv1 = ConvBNAct(cin, mid, k=1, s=1, p=0, act=act)
        self.conv2 = ConvBNAct(mid, mid, k=3, s=stride, p=1, act=act)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
        )

        self.se = SEBlock(cout) if use_se else nn.Identity()
        self.sd = StochasticDepth(sd_p)
        self.act = nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.se(out)

        out = self.sd(out)          # ✅ Stochastic Depth burada (branch)
        out = out + identity
        out = self.act(out)
        return out


# =========================
# 3) Model: ResNet-SE + StochasticDepth schedule
# =========================
def linear_sd_probs(total_blocks: int, sd_max: float):
    """Derinliğe göre lineer artan SD p listesi."""
    if total_blocks <= 1:
        return [sd_max]
    return [sd_max * i / (total_blocks - 1) for i in range(total_blocks)]


class ResNetSEStochasticDepth(nn.Module):
    """
    CIFAR (32x32) ve benzeri için iyi bir model.
    - Bottleneck + SE
    - Stage bazlı downsample
    - SD p değerleri block index’e göre lineer artar
    """
    def __init__(self, num_classes: int = 100, blocks=(2, 2, 2), widths=(64, 128, 256),
                 sd_max: float = 0.2, act: str = "silu", use_se: bool = True):
        super().__init__()
        assert len(blocks) == len(widths)

        self.stem = nn.Sequential(
            ConvBNAct(3, 64, k=3, s=1, p=1, act=act),
            ConvBNAct(64, 64, k=3, s=1, p=1, act=act),
        )

        total = sum(blocks)
        sd_probs = linear_sd_probs(total, sd_max)
        idx = 0

        in_ch = 64
        stages = []
        for stage_i, (n_blocks, out_ch) in enumerate(zip(blocks, widths)):
            stride = 1 if stage_i == 0 else 2
            for b in range(n_blocks):
                s = stride if b == 0 else 1
                p = sd_probs[idx]; idx += 1
                stages.append(BottleneckSD(in_ch, out_ch, stride=s, sd_p=p, use_se=use_se, act=act))
                in_ch = out_ch

        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# =========================
# 4) Quick test
# =========================
if __name__ == "__main__":
    model = ResNetSEStochasticDepth(num_classes=100, blocks=(2,2,2), widths=(64,128,256), sd_max=0.2)
    model.train()
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print("logits:", y.shape)  # [4, 100]