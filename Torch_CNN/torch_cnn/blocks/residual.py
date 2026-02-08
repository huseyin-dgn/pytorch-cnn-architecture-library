import torch
import torch.nn as nn

from torch_cnn.blocks.norm import make_norm

def _make_act(act: str):
    a = act.lower()
    if a == "relu":
        return nn.ReLU(inplace=True)
    if a == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError("act: 'relu' | 'silu'")

class PreActResidualConvBlock(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        stride: int = 1,
        norm: str = "bn",
        act: str = "relu",
        alpha: float = 1.0,
        skip_norm: bool = False,
    ):
        super().__init__()
        self.cin = int(cin)
        self.cout = int(cout)
        self.stride = int(stride)

        self.norm1 = make_norm(norm, self.cin)
        self.act1 = _make_act(act)

        self.conv1 = nn.Conv2d(self.cin, self.cout, 3, stride=self.stride, padding=1, bias=False)

        self.norm2 = make_norm(norm, self.cout)
        self.act2 = _make_act(act)
        self.conv2 = nn.Conv2d(self.cout, self.cout, 3, stride=1, padding=1, bias=False)

        self.proj = None
        if (self.cin != self.cout) or (self.stride != 1):
            layers = [nn.Conv2d(self.cin, self.cout, 1, stride=self.stride, bias=False)]
            if skip_norm:
                layers.append(make_norm(norm, self.cout))
            self.proj = nn.Sequential(*layers)

        self.register_buffer("alpha_const", torch.tensor(float(alpha)))

    def forward(self, x: torch.Tensor):
        skip = x if self.proj is None else self.proj(x)

        y = self.conv1(self.act1(self.norm1(x)))
        y = self.conv2(self.act2(self.norm2(y)))

        a = self.alpha_const.to(device=x.device, dtype=x.dtype)
        return skip + a * y
