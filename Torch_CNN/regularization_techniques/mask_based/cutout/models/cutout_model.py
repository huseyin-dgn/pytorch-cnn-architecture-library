import random
import torch
import torch.nn as nn

class AdvancedCutoutBlock2D(nn.Module):
    """
    Feature-map Cutout (B,C,H,W) üzerinde çalışır.

    Özellikler:
    - schedule: p ve size zamanla 0 → max’a çıkar
    - multi-hole: aynı forward’da birden fazla blok keser
    - fill: 'zero' veya 'mean'
    - per_sample: True => her sample farklı blok, False => batch aynı blok
    - channel_shared: True => tüm kanallara aynı uzamsal blok
    """
    def __init__(
        self,
        p_max: float = 0.2,
        size_min: int = 2,
        size_max: int = 8,
        holes: int = 2,
        total_steps: int = 10_000,
        fill: str = "zero",
        per_sample: bool = True,
        channel_shared: bool = True,
    ):
        super().__init__()
        if not (0.0 <= p_max < 1.0):
            raise ValueError("p_max [0,1) olmalı")
        if size_min < 1 or size_max < size_min:
            raise ValueError("size_min>=1 ve size_max>=size_min olmalı")
        if holes < 1:
            raise ValueError("holes >= 1 olmalı")
        if total_steps < 0:
            raise ValueError("total_steps >= 0 olmalı")
        if fill not in ["zero", "mean"]:
            raise ValueError("fill 'zero' veya 'mean' olmalı")

        self.p_max = float(p_max)
        self.size_min = int(size_min)
        self.size_max = int(size_max)
        self.holes = int(holes)
        self.total_steps = int(total_steps)
        self.fill = fill
        self.per_sample = bool(per_sample)
        self.channel_shared = bool(channel_shared)

        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def _progress(self) -> float:
        s = int(self.step.item())
        if self.total_steps <= 0:
            return 1.0
        t = s / float(self.total_steps)
        return max(0.0, min(1.0, t))

    def _current_p(self) -> float:
        return self.p_max * self._progress()

    def _current_size(self) -> int:
        t = self._progress()
        return int(round(self.size_min + (self.size_max - self.size_min) * t))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.step += 1

        if (not self.training):
            return x

        p = self._current_p()
        if p <= 0.0 or random.random() > p:
            return x

        if x.dim() != 4:
            raise ValueError(f"AdvancedCutoutBlock2D 4D bekler (B,C,H,W). Geldi: {tuple(x.shape)}")

        b, c, h, w = x.shape
        size = min(self._current_size(), h, w)

        # mean fill için sample başına mean
        mean = None
        if self.fill == "mean":
            mean = x.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)

        def apply_one(img: torch.Tensor, mean_img: torch.Tensor | None):
            # img: (C,H,W)
            for _ in range(self.holes):
                cy = random.randint(0, h - 1)
                cx = random.randint(0, w - 1)
                y1 = max(0, cy - size // 2)
                y2 = min(h, cy + (size - size // 2))
                x1 = max(0, cx - size // 2)
                x2 = min(w, cx + (size - size // 2))

                if self.fill == "zero":
                    img[:, y1:y2, x1:x2] = 0.0
                else:
                    # mean fill: channel-wise mean
                    img[:, y1:y2, x1:x2] = mean_img  # (C,1,1) broadcast
            return img

        if self.per_sample:
            # her sample ayrı blok
            for i in range(b):
                if self.fill == "mean":
                    apply_one(x[i], mean[i])
                else:
                    apply_one(x[i], None)
            return x

        # batch aynı blok (daha deterministik/az gürültü)
        cy = random.randint(0, h - 1)
        cx = random.randint(0, w - 1)
        y1 = max(0, cy - size // 2)
        y2 = min(h, cy + (size - size // 2))
        x1 = max(0, cx - size // 2)
        x2 = min(w, cx + (size - size // 2))

        if self.fill == "zero":
            x[:, :, y1:y2, x1:x2] = 0.0
        else:
            x[:, :, y1:y2, x1:x2] = mean  # (B,C,1,1) broadcast

        return x
    
import torch
import torch.nn as nn

class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, act="relu"):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlockCutout(nn.Module):
    def __init__(self, cin, cout, stride=1, act="relu", cutout: nn.Module | None = None):
        super().__init__()
        self.conv1 = ConvBNAct(cin, cout, 3, stride, 1, act)
        self.conv2 = nn.Sequential(
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.SiLU(inplace=True)
        self.cutout = cutout

        self.shortcut = nn.Identity()
        if cin != cout or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.act(out)          # ✅ activation sonrası
        if self.cutout is not None:  # ✅ cutout burada
            out = self.cutout(out)
        return out + self.shortcut(x)

class CIFARResNetWithCutout(nn.Module):
    def __init__(
        self,
        num_classes=100,
        act="relu",
        use_cutout=True,
        cutout_cfg=None,
    ):
        super().__init__()
        if cutout_cfg is None:
            cutout_cfg = dict(
                p_max=0.2,
                size_min=2,
                size_max=8,
                holes=2,
                total_steps=10_000,
                fill="zero",
                per_sample=True,
                channel_shared=True,
            )

        self.stem = nn.Sequential(
            ConvBNAct(3, 64, 3, 1, 1, act),
            ConvBNAct(64, 64, 3, 1, 1, act),
        )

        # Cutout'u stage2+stage3'e koymak genelde daha stabil
        cut = AdvancedCutoutBlock2D(**cutout_cfg) if use_cutout else None

        def make_stage(cin, cout, n, stride, cutout_here: bool):
            layers = []
            for i in range(n):
                s = stride if i == 0 else 1
                layers.append(
                    ResidualBlockCutout(
                        cin if i == 0 else cout,
                        cout,
                        stride=s,
                        act=act,
                        cutout=(cut if cutout_here else None),
                    )
                )
            return nn.Sequential(*layers)

        self.stage1 = make_stage(64, 64, n=3, stride=1, cutout_here=False)
        self.stage2 = make_stage(64, 128, n=3, stride=2, cutout_here=True)
        self.stage3 = make_stage(128, 256, n=3, stride=2, cutout_here=True)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = CIFARResNetWithCutout(num_classes=100, use_cutout=True)
model.train()
x = torch.randn(4, 3, 32, 32)
y = model(x)
print(y.shape)  # torch.Size([4, 100])