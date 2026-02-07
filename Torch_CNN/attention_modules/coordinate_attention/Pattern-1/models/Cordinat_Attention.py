import torch
import torch.nn as nn
import torch.nn.functional as F

class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=True) / 6.0


class CoordinateAtt(nn.Module):
    def __init__(
        self,
        in_channels: int,          # C: giriş kanal sayısı
        reduction: int = 32,       # bottleneck oranı (C // reduction)
        min_mid_channels: int = 8, # bottleneck alt sınırı
        act: str = "hswish",       # "hswish" veya "relu"
        alpha: float = 1.0,        # attention gücü (0: kapalı, 1: tam)
        learnable_alpha: bool = False, # True ise alpha öğrenilebilir
    ):
        super().__init__()

        mid_channels = max(min_mid_channels, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        if act.lower() == "hswish":
            self.act = HSwish()
        elif act.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError("act must be 'hswish' or 'relu'")

        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=True)  # H maskesi üretir
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=True)  # W maskesi üretir

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)))

        self._last_ah = None
        self._last_aw = None

    def forward(self, x):
        B, C, H, W = x.shape

        x_h = x.mean(dim=3, keepdim=True)                    # (B,C,H,1)  -> W üzerinde ortalama (H ekseni bilgisi kalır)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2) # (B,C,W,1)  -> H üzerinde ortalama, concat için permute

        y = torch.cat([x_h, x_w], dim=2)                     # (B,C,H+W,1)

        y = self.act(self.bn1(self.conv1(y)))                # (B,mid,H+W,1)

        y_h, y_w = torch.split(y, [H, W], dim=2)             # y_h:(B,mid,H,1), y_w:(B,mid,W,1)
        y_w = y_w.permute(0, 1, 3, 2)                        # (B,mid,1,W)

        a_h = torch.sigmoid(self.conv_h(y_h))                # (B,C,H,1)
        a_w = torch.sigmoid(self.conv_w(y_w))                # (B,C,1,W)

        self._last_ah = a_h
        self._last_aw = a_w

        att = a_h * a_w                                      # (B,C,H,W)  -> broadcast ile birleşir
        scale = (1.0 - self.alpha) + self.alpha * att        # alpha ile kontrol

        return x * scale

    @torch.no_grad()
    def last_mask_stats(self):
        if self._last_ah is None or self._last_aw is None:
            return None
        ah = self._last_ah
        aw = self._last_aw
        return {
            "a_h": {"min": float(ah.min()), "mean": float(ah.mean()), "max": float(ah.max()), "std": float(ah.std())},
            "a_w": {"min": float(aw.min()), "mean": float(aw.mean()), "max": float(aw.max()), "std": float(aw.std())},
        }