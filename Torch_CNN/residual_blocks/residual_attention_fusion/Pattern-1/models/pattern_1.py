from __future__ import annotations
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) -> (B,C,1,1)
        return self.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.max(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) -> (B,1,H,W)
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_map, max_map], dim=1)  # (B,2,H,W)
        return self.sigmoid(self.conv(cat))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class BasicResCBAMBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ):
        super().__init__()
        self.act = nn.ReLU(inplace=True)

        # residual branch
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.cbam = CBAM(out_ch, reduction=reduction, spatial_kernel=spatial_kernel)

        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.skip is None else self.skip(x)

        f = self.act(self.bn1(self.conv1(x)))
        f = self.bn2(self.conv2(f))  # F(x)

        f = self.cbam(f)             # A(F(x)) ⊙ F(x)

        y = identity + f
        y = self.act(y)
        return y


class BottleneckResCBAMBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        mid = out_ch

        self.conv1 = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)

        self.conv2 = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)

        self.conv3 = nn.Conv2d(mid, out_ch * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion)

        self.cbam = CBAM(out_ch * self.expansion, reduction=reduction, spatial_kernel=spatial_kernel)

        self.skip = None
        out_full = out_ch * self.expansion
        if stride != 1 or in_ch != out_full:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_full, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_full),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.skip is None else self.skip(x)

        f = self.act(self.bn1(self.conv1(x)))
        f = self.act(self.bn2(self.conv2(f)))
        f = self.bn3(self.conv3(f))      # F(x)

        f = self.cbam(f)                 # A(F(x)) ⊙ F(x)

        y = identity + f
        y = self.act(y)
        return y

class ResCBAMNet(nn.Module):

    def __init__(
        self,
        num_classes: int = 10,
        block_type: str = "basic",
        base_ch: int = 64,
        reduction: int = 16,
        spatial_kernel: int = 7,
        dropout_p: float = 0.0,
    ):
        super().__init__()

        if block_type == "basic":
            Block = BasicResCBAMBlock
            ch1, ch2, ch3 = base_ch, base_ch * 2, base_ch * 4
            out_head = ch3
        elif block_type == "bottleneck":
            Block = BottleneckResCBAMBlock
            ch1, ch2, ch3 = base_ch, base_ch * 2, base_ch * 4
            out_head = ch3 * Block.expansion
        else:
            raise ValueError("block_type must be 'basic' or 'bottleneck'")

        self.stem = nn.Sequential(
            nn.Conv2d(3, ch1, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True),
        )

        self.b1 = Block(ch1, ch1, stride=1, reduction=reduction, spatial_kernel=spatial_kernel)
        self.b2 = Block(ch1, ch1, stride=1, reduction=reduction, spatial_kernel=spatial_kernel)

        self.b3 = Block(ch1, ch2, stride=2, reduction=reduction, spatial_kernel=spatial_kernel)
        self.b4 = Block(ch2 if block_type == "basic" else ch2 * Block.expansion, ch2, stride=1,
                        reduction=reduction, spatial_kernel=spatial_kernel)

        in_b5 = ch2 if block_type == "basic" else ch2 * Block.expansion
        self.b5 = Block(in_b5, ch3, stride=2, reduction=reduction, spatial_kernel=spatial_kernel)
        self.b6 = Block(ch3 if block_type == "basic" else ch3 * Block.expansion, ch3, stride=1,
                        reduction=reduction, spatial_kernel=spatial_kernel)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.fc = nn.Linear(out_head, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x = self.b1(x)
        x = self.b2(x)

        x = self.b3(x)
        x = self.b4(x)

        x = self.b5(x)
        x = self.b6(x)

        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    x = torch.randn(4, 3, 32, 32)

    m_basic = ResCBAMNet(num_classes=10, block_type="basic", base_ch=64)
    y_basic = m_basic(x)
    print("basic:", y_basic.shape)

    m_bneck = ResCBAMNet(num_classes=10, block_type="bottleneck", base_ch=32)
    y_bneck = m_bneck(x)
    print("bottleneck:", y_bneck.shape)