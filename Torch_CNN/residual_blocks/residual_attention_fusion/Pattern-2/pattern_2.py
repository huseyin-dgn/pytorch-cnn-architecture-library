from __future__ import annotations
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channel:int,reduction:int =16):
        super().__init__()
        hidden = max(channel // reduction , 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                nn.Conv2d(channel,hidden,1,bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden,channel,1,bias=False))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        return self.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.max(x)))
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel:int = 7):
        super().__init__()
        assert kernel in (3,7)
        padding = 3 if kernel == 7 else 1
        self.conv = nn.Conv2d(2,1,kernel,padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        a = torch.mean(x,dim=1,keepdim=True)
        m = torch.max(x,dim=1,keepdim=True)
        a_m = torch.cat([a,m],dim=1)
        return self.sigmoid(self.conv(a_m))

class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class ControlledP2Block(nn.Module):
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
        self.act = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_ch,out_ch,3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch,out_ch,3,stride=1,padding=1,bias=False)

        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                    nn.Conv2d(in_ch,out_ch,1,bias=False),
                    nn.BatchNorm2d(out_ch))
        
        self.cbam = CBAM(in_ch,out_ch,reduction=reduction,spatial_kernel=spatial_kernel)
        
        lam_init = float(lam_init)
        lam_init = min(max(lam_init,1e-4),1-1e-4)
        lam_logit = torch.log(torch.tensor(lam_init)/ (1 - torch.tensor(lam_init)))
        
        if lam_learnable:
            self.lam_logit = nn.Parameter
        else:
            self.register_buffer("lam_logit",lam_logit)

    def forward(self,x):
        identity = x if self.skip is None else self.skip(x)

        f = self.act(self.bn1(self.conv1(x)))
        f = self.bn2(self.conv2(x))

        z = identity + f
        z_att = self.cbam(z)

        lam = torch.sigmoid(self.lam_logit)
        y = (1.0 - lam) * z + lam * z_att

        return self.act(y)

class SimpleCNN_ControlledP2(nn.Module):
    def __init__(self, num_classes: int = 10, lam_init: float = 0.1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Controlled Pattern-2 blocks
        self.p2_1 = ControlledP2Block(64, 64, stride=1, lam_init=lam_init)
        self.p2_2 = ControlledP2Block(64, 128, stride=2, lam_init=lam_init)

        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)

        x = self.p2_1(x)
        x = self.p2_2(x)

        x = self.block2(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = SimpleCNN_ControlledP2(num_classes=10, lam_init=0.1)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print("Output:", y.shape)  # (4, 10)

    print("lambda p2_1:", torch.sigmoid(model.p2_1.lam_logit).item())
    print("lambda p2_2:", torch.sigmoid(model.p2_2.lam_logit).item())