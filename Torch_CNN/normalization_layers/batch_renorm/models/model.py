# brn_model.py
import torch
import torch.nn as nn

class BatchRenorm2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        rmax: float = 3.0,
        dmax: float = 5.0,
        warmup_steps: int = 5000,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.rmax_target = float(rmax)
        self.dmax_target = float(dmax)
        self.warmup_steps = int(warmup_steps)

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def _current_clamp(self):
        if not self.track_running_stats:
            return self.rmax_target, self.dmax_target

        t = int(self.num_batches_tracked.item())
        alpha = 1.0 if self.warmup_steps <= 0 else min(1.0, t / self.warmup_steps)
        rmax = 1.0 + alpha * (self.rmax_target - 1.0)
        dmax = 0.0 + alpha * (self.dmax_target - 0.0)
        return rmax, dmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("BatchRenorm2d expects (N, C, H, W)")

        if self.track_running_stats:
            self.num_batches_tracked += 1

        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            std = torch.sqrt(var + self.eps)

            if self.track_running_stats:
                rstd = torch.sqrt(self.running_var + self.eps)

                r = (std / rstd).detach()
                d = ((mean - self.running_mean) / rstd).detach()

                rmax, dmax = self._current_clamp()
                r = torch.clamp(r, 1.0 / rmax, rmax)
                d = torch.clamp(d, -dmax, dmax)
            else:
                r = 1.0
                d = 0.0

            x_hat = (x - mean[None, :, None, None]) / std[None, :, None, None]

            if self.track_running_stats:
                x_hat = x_hat * r[None, :, None, None] + d[None, :, None, None]

                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            if not self.track_running_stats:
                raise RuntimeError("Eval requires track_running_stats=True")
            x_hat = (x - self.running_mean[None, :, None, None]) / torch.sqrt(
                self.running_var[None, :, None, None] + self.eps
            )

        if self.affine:
            x_hat = x_hat * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return x_hat


class ConvBRNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, rmax=3.0, dmax=5.0, warmup_steps=5000):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.brn  = BatchRenorm2d(out_ch, rmax=rmax, dmax=dmax, warmup_steps=warmup_steps)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.brn(self.conv(x)))

class ResidualBRNBlock(nn.Module):
    def __init__(self, ch, rmax=3.0, dmax=5.0, warmup_steps=5000):
        super().__init__()
        self.f1 = ConvBRNAct(ch, ch, 3, 1, 1, rmax=rmax, dmax=dmax, warmup_steps=warmup_steps)
        self.f2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            BatchRenorm2d(ch, rmax=rmax, dmax=dmax, warmup_steps=warmup_steps),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.f2(self.f1(x)))

class BRNBackbone(nn.Module):
    def __init__(self, width=32, rmax=3.0, dmax=5.0, warmup_steps=5000):
        super().__init__()
        w = width
        self.stem = ConvBRNAct(3, w, 3, 2, 1, rmax, dmax, warmup_steps)        # /2

        self.stage1 = nn.Sequential(
            ConvBRNAct(w, w*2, 3, 2, 1, rmax, dmax, warmup_steps),             # /4
            ResidualBRNBlock(w*2, rmax, dmax, warmup_steps),
            ResidualBRNBlock(w*2, rmax, dmax, warmup_steps),
        )
        self.stage2 = nn.Sequential(
            ConvBRNAct(w*2, w*4, 3, 2, 1, rmax, dmax, warmup_steps),           # /8
            ResidualBRNBlock(w*4, rmax, dmax, warmup_steps),
            ResidualBRNBlock(w*4, rmax, dmax, warmup_steps),
        )
        self.stage3 = nn.Sequential(
            ConvBRNAct(w*4, w*8, 3, 2, 1, rmax, dmax, warmup_steps),           # /16
            ResidualBRNBlock(w*8, rmax, dmax, warmup_steps),
            ResidualBRNBlock(w*8, rmax, dmax, warmup_steps),
        )

    def forward(self, x, return_features=False):
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        if return_features:
            return f1, f2, f3
        return f3

class BRNClassifier(nn.Module):
    def __init__(self, num_classes=10, width=32, rmax=3.0, dmax=5.0, warmup_steps=5000):
        super().__init__()
        self.backbone = BRNBackbone(width=width, rmax=rmax, dmax=dmax, warmup_steps=warmup_steps)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width * 8, num_classes)

    def forward(self, x):
        feat = self.backbone(x)                 # (N, C, H, W)
        feat = self.pool(feat).flatten(1)       # (N, C)
        return self.fc(feat)

if __name__ == "__main__":
    model = BRNClassifier(num_classes=10, width=16, warmup_steps=0)
    model.train()

    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("logits:", y.shape)

    f1, f2, f3 = model.backbone(x, return_features=True)
    print("f1,f2,f3:", f1.shape, f2.shape, f3.shape)