import torch
import torch.nn as nn
import torch.nn.functional as F

class FrozenBatchNorm2d(nn.Module):
    # BN gibi normalize eder ama running stats güncellemez (batch'ten yeni istatistik almaz)
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)

        # donmuş istatistikler (BN'den kopyalanacak)
        self.register_buffer("running_mean", torch.zeros(self.num_features))
        self.register_buffer("running_var", torch.ones(self.num_features))

        if self.affine:
            # gamma / beta (BN weight/bias'tan kopyalanacak)
            self.register_buffer("weight", torch.ones(self.num_features))
            self.register_buffer("bias", torch.zeros(self.num_features))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.running_mean.view(1, -1, 1, 1)
        var = self.running_var.view(1, -1, 1, 1)

        if self.affine:
            w = self.weight.view(1, -1, 1, 1)
            b = self.bias.view(1, -1, 1, 1)
        else:
            w = 1.0
            b = 0.0

        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * w + b

class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1, act=True, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if p is None:
            p = k // 2

        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=k, stride=s, padding=p,
            groups=groups, bias=False
        )
        # burada BN ile başlayacağız (A yöntemi)
        self.norm = norm_layer(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expansion=4, se_ratio=0.25, norm_layer=nn.BatchNorm2d):
        super().__init__()
        mid_ch = out_ch // expansion
        assert out_ch % expansion == 0

        self.conv1 = ConvNormAct(in_ch, mid_ch, k=1, s=1, p=0, norm_layer=norm_layer)
        self.conv2 = ConvNormAct(mid_ch, mid_ch, k=3, s=stride, p=1, norm_layer=norm_layer)
        self.conv3 = ConvNormAct(mid_ch, out_ch, k=1, s=1, p=0, act=False, norm_layer=norm_layer)

        self.se = SEModule(out_ch, reduction=int(1 / se_ratio)) if se_ratio and se_ratio > 0 else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, k=1, s=stride, p=0, act=False, norm_layer=norm_layer)
        else:
            self.shortcut = nn.Identity()

        self.out_act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv3(self.conv2(self.conv1(x)))
        out = self.se(out)
        out = out + identity
        return self.out_act(out)

class Net(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=1000,
        widths=(64, 128, 256, 512),
        depths=(2, 2, 6, 2),
        stem_width=64,
        norm_layer=nn.BatchNorm2d,  # başlangıçta BN
    ):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, stem_width, k=3, s=2, p=1, norm_layer=norm_layer),
            ConvNormAct(stem_width, stem_width, k=3, s=1, p=1, norm_layer=norm_layer),
            ConvNormAct(stem_width, widths[0], k=3, s=1, p=1, norm_layer=norm_layer),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self.make_stage(widths[0], widths[0], depths[0], stride=1, norm_layer=norm_layer)
        self.stage2 = self.make_stage(widths[0], widths[1], depths[1], stride=2, norm_layer=norm_layer)
        self.stage3 = self.make_stage(widths[1], widths[2], depths[2], stride=2, norm_layer=norm_layer)
        self.stage4 = self.make_stage(widths[2], widths[3], depths[3], stride=2, norm_layer=norm_layer)

        self.head_norm = norm_layer(widths[3])  # BN ile train edeceğiz
        self.classifier = nn.Linear(widths[3], num_classes)

    def make_stage(self, in_ch, out_ch, depth, stride, norm_layer):
        blocks = []
        for i in range(depth):
            s = stride if i == 0 else 1
            blocks.append(
                Bottleneck(
                    in_ch=in_ch if i == 0 else out_ch,
                    out_ch=out_ch,
                    stride=s,
                    expansion=4,
                    se_ratio=0.25,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head_norm(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)

def convert_bn_to_frozen(module: nn.Module):
    for name, child in list(module.named_children()):
        # Alt modülleri önce dolaş (recursive)
        convert_bn_to_frozen(child)
        # Eğer bu child bir BatchNorm2d ise, FrozenBN ile değiştir
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(
                num_features=child.num_features,
                eps=child.eps,
                affine=child.affine
            )
            # running stats kopyala
            frozen.running_mean.copy_(child.running_mean.detach())
            frozen.running_var.copy_(child.running_var.detach())
            # affine varsa gamma/beta kopyala
            if child.affine:
                frozen.weight.copy_(child.weight.detach())
                frozen.bias.copy_(child.bias.detach())

            # modülde replace et
            setattr(module, name, frozen)

if __name__ == "__main__":
    # 1) Train edilecek model (BatchNorm2d ile)
    model = Net(num_classes=10, norm_layer=nn.BatchNorm2d)
    # 2) BN -> FrozenBN çevir
    convert_bn_to_frozen(model)
    # 3) Artık model FrozenBN kullanıyor (BN güncellenmeyecek)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output:", y.shape)  # (2, 10)