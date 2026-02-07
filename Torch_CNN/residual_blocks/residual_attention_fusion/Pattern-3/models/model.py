import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.max(x)))  # (B,C,1,1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)          # (B,1,H,W)
        max_map, _ = torch.max(x, dim=1, keepdim=True)        # (B,1,H,W)
        cat = torch.cat([avg_map, max_map], dim=1)            # (B,2,H,W)
        return self.sigmoid(self.conv(cat))                   # (B,1,H,W)


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class FxConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.act(self.bn1(self.conv1(x)))
        f = self.bn2(self.conv2(f))
        return f


def make_skip(in_ch: int, out_ch: int, stride: int):
    if stride != 1 or in_ch != out_ch:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
    return nn.Identity()

class Pattern3Block(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        reduction: int = 16,
        spatial_kernel: int = 7,
        lam_init: float = 0.1,
        lam_learnable: bool = True,
    ):
        super().__init__()
        self.attn = CBAM(in_ch, reduction=reduction, spatial_kernel=spatial_kernel)
        self.F = FxConv(in_ch, out_ch, stride=stride)
        self.skip = make_skip(in_ch, out_ch, stride)
        self.out_act = nn.ReLU(inplace=True)

        lam_init = float(lam_init)
        lam_init = min(max(lam_init, 1e-4), 1 - 1e-4)
        lam_logit = torch.log(torch.tensor(lam_init) / (1 - torch.tensor(lam_init)))
        if lam_learnable:
            self.lam_logit = nn.Parameter(lam_logit)
        else:
            self.register_buffer("lam_logit", lam_logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        x_att = self.attn(x)                    # CBAM(x) = A(x) ⊙ x
        lam = torch.sigmoid(self.lam_logit)     # scalar (0,1)
        x_tilde = (1.0 - lam) * x + lam * x_att 

        f = self.F(x_tilde)
        y = identity + f
        return self.out_act(y)

class Pattern3Net(nn.Module):
    def __init__(self, num_classes: int = 10, lam_init: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Pattern3Block(64, 64, stride=1, lam_init=lam_init),
            Pattern3Block(64, 64, stride=1, lam_init=lam_init),
        )

        self.stage2 = nn.Sequential(
            Pattern3Block(64, 128, stride=2, lam_init=lam_init), 
            Pattern3Block(128, 128, stride=1, lam_init=lam_init),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.head(x)
        return x

if __name__ == "__main__":
    m = Pattern3Net(num_classes=10, lam_init=0.1)
    x = torch.randn(4, 3, 32, 32)
    y = m(x)
    print("out:", y.shape)

    for name, mod in m.named_modules():
        if isinstance(mod, Pattern3Block):
            print(name, "lambda=", float(torch.sigmoid(mod.lam_logit)))