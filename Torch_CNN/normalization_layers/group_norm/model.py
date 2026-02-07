import torch
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
        mode: str = "gn"
    ):
        assert mode in {"gn", "ln_like", "in_like"}
        self.groups = groups
        self.eps = eps
        self.affine = affine
        self.preferred_groups = preferred_groups
        self.mode = mode


class GroupNormFlex(nn.Module):
    def __init__(self, num_channels: int, config: GNConfig | None = None):
        # # config parametresi ya bir GNConfig nesnesi olur ya da None olur.
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


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        act: str = "silu",
        gn_config: GNConfig | None = None,
    ):
        super().__init__()
        if p is None:
            p = k // 2

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.norm = GroupNormFlex(out_ch, gn_config)

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "gelu":
            self.act = nn.GELU()
        elif act in ("none", None):
            self.act = nn.Identity()
        else:
            raise ValueError("act: 'relu' | 'silu' | 'gelu' | 'none'")

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class GNCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        base_channels: int = 32,
        gn_config: GNConfig | None = None,
        act: str = "silu",
        return_features: bool = False
    ):
        super().__init__()
        self.return_features = return_features
        if gn_config is None:
            gn_config = GNConfig(groups="auto", mode="gn")

        C1 = base_channels
        C2 = base_channels * 2
        C3 = base_channels * 4

        # Stem: H,W -> H/2,W/2
        self.stem = nn.Sequential(
            ConvGNAct(3, C1, k=3, s=2, act=act, gn_config=gn_config),
            ConvGNAct(C1, C1, k=3, s=1, act=act, gn_config=gn_config),
        )

        # Stage 1: H/2 -> H/4
        self.stage1 = nn.Sequential(
            ConvGNAct(C1, C2, k=3, s=2, act=act, gn_config=gn_config),
            ConvGNAct(C2, C2, k=3, s=1, act=act, gn_config=gn_config),
        )

        # Stage 2: H/4 -> H/8
        self.stage2 = nn.Sequential(
            ConvGNAct(C2, C3, k=3, s=2, act=act, gn_config=gn_config),
            ConvGNAct(C3, C3, k=3, s=1, act=act, gn_config=gn_config),
        )

        # Head (classification)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(C3, num_classes)

    def forward(self, x):
        # x: (B,3,H,W)
        p3 = self.stem(x)      # (B, C1, H/2, W/2)
        p4 = self.stage1(p3)   # (B, C2, H/4, W/4)
        p5 = self.stage2(p4)   # (B, C3, H/8, W/8)

        logits = self.fc(self.pool(p5).flatten(1))

        if self.return_features:
            return p3, p4, p5, logits
        return logits

if __name__ == "__main__":
    cfg = GNConfig(groups="auto", mode="gn")   
    model = GNCNN(num_classes=10, base_channels=32, gn_config=cfg, return_features=True)

    x = torch.randn(4, 3, 256, 256)
    p3, p4, p5, logits = model(x)

    print("p3:", p3.shape)       # (4, 32, 128, 128)
    print("p4:", p4.shape)       # (4, 64, 64, 64)
    print("p5:", p5.shape)       # (4, 128, 32, 32)
    print("logits:", logits.shape)  # (4, 10)