import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock2d(nn.Module):
    def __init__(self, block_size: int = 7, total_steps: int = 10_000, max_drop_prob: float = 0.1):
        super().__init__()
        self.block_size = int(block_size)
        self.total_steps = int(total_steps)
        self.max_drop_prob = float(max_drop_prob)
        self.register_buffer("step", torch.zeros((), dtype=torch.long))  # scalar

    def _current_drop_prob(self) -> float:
        s = int(self.step.item())
        t = min(max(s / max(1, self.total_steps), 0.0), 1.0)
        return self.max_drop_prob * t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.step += 1

        drop_prob = self._current_drop_prob()
        if (not self.training) or drop_prob <= 0.0:
            return x

        if x.dim() != 4:
            raise ValueError(f"DropBlock 4D tensor bekler (N,C,H,W). Geldi: {x.shape}")

        n, c, h, w = x.shape
        bs = min(self.block_size, h, w)
        if bs < 1:
            return x

        valid_h = h - bs + 1
        valid_w = w - bs + 1
        block_area = bs * bs

        gamma = drop_prob * (h * w) / (block_area * valid_h * valid_w)

        center_mask = (torch.rand((n, c, valid_h, valid_w), device=x.device) < gamma).float()

        pad_h = (bs - 1) // 2
        pad_w = (bs - 1) // 2
        center_mask = F.pad(center_mask, (pad_w, pad_w, pad_h, pad_h))

        block_mask = F.max_pool2d(center_mask, kernel_size=bs, stride=1, padding=bs // 2)
        mask = 1.0 - block_mask  # keep=1, drop=0

        keep_prob = mask.mean().clamp(min=1e-5)
        return x * mask / keep_prob
    
def _get_act(act: str) -> nn.Module:
    a = act.lower()
    if a == "relu":
        return nn.ReLU(inplace=True)
    if a in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if a == "gelu":
        return nn.GELU()
    raise ValueError("act: 'relu' | 'silu' | 'gelu'")

class ConvBNAct(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        act: str = "relu",
        dropblock: nn.Module | None = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = _get_act(act)
        self.dropblock = dropblock

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.conv(x)))
        if self.dropblock is not None:
            x = self.dropblock(x)
        return x

class BasicResBlock(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        stride: int = 1,
        act: str = "relu",
        dropblock: nn.Module | None = None,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(cin, cout, k=3, s=stride, p=1, act=act, dropblock=None)  # genelde ilk conv'a koyma
        self.conv2 = nn.Sequential(
            nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.act = _get_act(act)
        self.dropblock = dropblock

        self.skip = nn.Identity()
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.act(out + identity)
        if self.dropblock is not None:
            out = self.dropblock(out)
        return out

class DropBlockResNetSmall(nn.Module):
    def __init__(
        self,
        num_classes: int,
        act: str = "relu",
        use_dropblock: bool = False,
        block_size: int = 7,
        max_drop_prob: float = 0.1,
        total_steps: int = 10_000,
    ):
        super().__init__()

        db = None
        if use_dropblock:
            db = DropBlock2d(block_size=block_size, total_steps=total_steps, max_drop_prob=max_drop_prob)

        # Stem
        self.stem = ConvBNAct(3, 64, k=3, s=1, p=1, act=act, dropblock=None)

        # Stage1: 32->32 (DropBlock yok)
        self.stage1 = nn.Sequential(
            BasicResBlock(64, 64, stride=1, act=act, dropblock=None),
            BasicResBlock(64, 64, stride=1, act=act, dropblock=None),
        )

        # Stage2: 32->16 (DropBlock var)
        self.stage2 = nn.Sequential(
            BasicResBlock(64, 128, stride=2, act=act, dropblock=db),
            BasicResBlock(128, 128, stride=1, act=act, dropblock=db),
        )

        # Stage3: 16->8 (DropBlock var)
        self.stage3 = nn.Sequential(
            BasicResBlock(128, 256, stride=2, act=act, dropblock=db),
            BasicResBlock(256, 256, stride=1, act=act, dropblock=db),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

        # küçük init dokunuşu (opsiyonel ama stabil)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)

def build_models(
    num_classes: int,
    train_loader_len: int,
    epochs: int = 15,
    act: str = "relu",
    block_size: int = 7,
    max_drop_prob: float = 0.1,
):
    total_steps = epochs * train_loader_len

    baseline = DropBlockResNetSmall(
        num_classes=num_classes,
        act=act,
        use_dropblock=False,
        block_size=block_size,
        max_drop_prob=max_drop_prob,
        total_steps=total_steps,
    )

    dropblock = DropBlockResNetSmall(
        num_classes=num_classes,
        act=act,
        use_dropblock=True,
        block_size=block_size,
        max_drop_prob=max_drop_prob,
        total_steps=total_steps,
    )

    return baseline, dropblock