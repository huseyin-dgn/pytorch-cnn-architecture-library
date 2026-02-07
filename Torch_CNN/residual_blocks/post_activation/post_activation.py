import torch
import torch.nn as nn

class BasicResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm="bn", attn=None):
        super().__init__()
        if norm == "bn":
            N = nn.BatchNorm2d
        elif norm == "gn":
            def N(c):
                g = min(32, c)
                while c % g != 0 and g > 2:
                    g //= 2
                if c % g != 0:
                    g = 2 if (c % 2 == 0) else 1
                return nn.GroupNorm(g, c)
        else:
            raise ValueError("norm 'bn' veya 'gn' olmalı.")

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = N(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.norm2 = N(out_ch)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                N(out_ch)
            )

        self.attn = attn

    def forward(self, x):
        skip = x

        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))

        if self.attn is not None:
            out = self.attn(out)

        if self.downsample is not None:
            skip = self.downsample(skip)

        out = self.relu(out + skip)
        return out


class MiniResNet(nn.Module):
    def __init__(
        self,
        num_classes=10,
        base_ch=64,
        layers=(2, 2, 2, 2),
        norm="bn",
        attn_factory=None,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch) if norm == "bn" else self._gn(base_ch),
            nn.ReLU(inplace=False),
        )

        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        self.stage1 = self._make_stage(chs[0], chs[0], layers[0], stride=1, norm=norm, attn_factory=attn_factory, stage_idx=1)
        self.stage2 = self._make_stage(chs[0], chs[1], layers[1], stride=2, norm=norm, attn_factory=attn_factory, stage_idx=2)
        self.stage3 = self._make_stage(chs[1], chs[2], layers[2], stride=2, norm=norm, attn_factory=attn_factory, stage_idx=3)
        self.stage4 = self._make_stage(chs[2], chs[3], layers[3], stride=2, norm=norm, attn_factory=attn_factory, stage_idx=4)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(chs[3], num_classes)
        )

    def _gn(self, c):
        g = min(32, c)
        while c % g != 0 and g > 2:
            g //= 2
        if c % g != 0:
            g = 2 if (c % 2 == 0) else 1
        return nn.GroupNorm(g, c)

    def _make_stage(self, in_ch, out_ch, n_blocks, stride, norm, attn_factory, stage_idx):
        blocks = []
        for i in range(n_blocks):
            s = stride if i == 0 else 1
            ch_in = in_ch if i == 0 else out_ch
            attn = None
            if attn_factory is not None:
                attn = attn_factory(out_ch, stage_idx=stage_idx, block_idx=i)
            blocks.append(BasicResidualBlock(ch_in, out_ch, stride=s, norm=norm, attn=attn))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)

model = MiniResNet(num_classes=10, norm="bn", attn_factory=None)
x = torch.randn(2, 3, 32, 32)
y = model(x)
print(y.shape)