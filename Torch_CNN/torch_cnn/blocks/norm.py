import torch.nn as nn

def choose_gn_groups(C: int, preferred=(32, 16, 8, 4, 2, 1)) -> int:
    for g in preferred:
        if g <= C and (C % g == 0):
            return g
    return 1

class GNConfig:
    def __init__(
        self,
        groups="auto",
        eps: float = 1e-5,
        affine: bool = True,
        preferred_groups=(32, 16, 8, 4, 2, 1),
        mode: str = "gn",
    ):
        if mode not in {"gn", "ln_like", "in_like"}:
            raise ValueError("mode: 'gn' | 'ln_like' | 'in_like'")
        self.groups = groups
        self.eps = eps
        self.affine = affine
        self.preferred_groups = preferred_groups
        self.mode = mode

class GroupNormFlex(nn.Module):
    def __init__(self, num_channels: int, config: GNConfig | None = None):
        super().__init__()
        if config is None:
            config = GNConfig()

        C = int(num_channels)

        if config.mode == "ln_like":
            groups = 1
        elif config.mode == "in_like":
            groups = C
        else:
            if config.groups == "auto":
                groups = choose_gn_groups(C, config.preferred_groups)
            else:
                groups = int(config.groups)

        if groups < 1:
            raise ValueError("num_groups >= 1 olmalı")
        if C % groups != 0:
            raise ValueError(f"C % G == 0 olmalı. C={C}, G={groups}")

        self.groups = groups
        self.gn = nn.GroupNorm(groups, C, eps=float(config.eps), affine=bool(config.affine))

    def forward(self, x):
        return self.gn(x)

def make_norm(norm: str, ch: int, *, gn_config: GNConfig | None = None) -> nn.Module:
    n = (norm or "none").lower()
    if n == "bn":
        return nn.BatchNorm2d(ch)
    if n == "gn":
        g = min(32, ch)
        while ch % g != 0 and g > 2:
            g //= 2
        if ch % g != 0:
            g = 2 if (ch % 2 == 0) else 1
        return nn.GroupNorm(g, ch)
    if n == "gnflex":
        return GroupNormFlex(ch, gn_config)
    if n == "none":
        return nn.Identity()
    raise ValueError("norm: 'none' | 'bn' | 'gn' | 'gnflex'")