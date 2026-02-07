import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# GhostConv: Primary + Cheap (depthwise) + Concat
# -------------------------------------------------
class GhostConv(nn.Module):
    """
    Ghost Convolution:
    - Primary conv: C_in -> C_int  (pahalı kısım)
    - Cheap conv (depthwise): C_int -> C_ghost
    - Concat: [primary, ghost] -> C_out
    """
    def __init__(self, c_in, c_out, kernel_size=1, ratio=2, dw_kernel_size=3,
                 stride=1, padding=0):
        super().__init__()
        self.c_out = c_out
        self.ratio = ratio

        # Dahili kanal sayısı: kaç tane "gerçek" feature üreteceğiz?
        c_int = int(round(c_out / ratio))
        self.c_int = c_int
        c_ghost = c_out - c_int

        # 1) Primary conv (pahalı kısım)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_int,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(c_int),
            nn.ReLU(inplace=True)
        )

        # 2) Cheap operation (ghost feature üretimi) - depthwise conv
        if c_ghost > 0:
            self.cheap_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=c_int,
                    out_channels=c_ghost,
                    kernel_size=dw_kernel_size,
                    stride=1,
                    padding=dw_kernel_size // 2,
                    groups=c_int,   # depthwise
                    bias=False
                ),
                nn.BatchNorm2d(c_ghost),
                nn.ReLU(inplace=True)
            )
        else:
            self.cheap_conv = None

    def forward(self, x):
        # 1) Temel feature'lar
        x_primary = self.primary_conv(x)   # (N, c_int, H', W')

        # 2) Ghost feature'lar
        if self.cheap_conv is not None:
            x_ghost = self.cheap_conv(x_primary)           # (N, c_ghost, H', W')
            out = torch.cat([x_primary, x_ghost], dim=1)   # (N, c_int + c_ghost, H', W')
        else:
            out = x_primary

        # 3) Fazlalığı kırp (round kaynaklı edge-case için)
        if out.size(1) > self.c_out:
            out = out[:, :self.c_out, :, :]

        return out


# -------------------------------------------------
# GhostBlock: GhostConv + (opsiyonel) residual
# -------------------------------------------------
class GhostBlock(nn.Module):
    """
    GhostConv tabanlı bir blok:
    - GhostConv(c_in -> c_out, kernel_size=3)
    - BN + aktivasyon GhostConv içinde zaten var
    - (İsteğe bağlı) residual bağlantı:
        * stride == 1 ve c_in == c_out ise identity
        * aksi halde 1x1 conv ile kısayol
    """
    def __init__(self, c_in, c_out, stride=1, ratio=2):
        super().__init__()
        self.stride = stride
        self.c_in = c_in
        self.c_out = c_out

        # Ana yol: GhostConv ile uzamsal + kanal işlemi
        self.ghost = GhostConv(
            c_in=c_in,
            c_out=c_out,
            kernel_size=3,      # spatial + channel
            ratio=ratio,
            dw_kernel_size=3,
            stride=stride,
            padding=1
        )

        # Shortcut tanımı
        if stride == 1 and c_in == c_out:
            # Basit identity
            self.shortcut = nn.Identity()
        else:
            # Boyut veya kanal farkını telafi etmek için 1x1 conv
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out)
            )

    def forward(self, x):
        out = self.ghost(x)
        shortcut = self.shortcut(x)
        out = out + shortcut
        # GhostConv içinde zaten ReLU var; burada ekstra aktivasyon opsiyonel
        out = F.relu(out, inplace=True)
        return out


# -------------------------------------------------
# GhostConv tabanlı küçük CNN (CIFAR-10 gibi)
# -------------------------------------------------
class GhostNetSmall(nn.Module):
    """
    Giriş: (N, 3, 32, 32)

    Yapı:
    - Stem: 3x3 Conv (3 -> 16)
    - Stage1: GhostBlock(16->16) x2  (32x32'de)
    - Stage2: GhostBlock(16->32, stride=2) + GhostBlock(32->32)  (16x16'da)
    - Stage3: GhostBlock(32->64, stride=2) + GhostBlock(64->64)  (8x8'de)
    - GAP + FC
    """
    def __init__(self, num_classes=10, ratio=2):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )  # (N, 16, 32, 32)

        # Stage 1: boyut sabit, kanal 16
        self.stage1 = nn.Sequential(
            GhostBlock(c_in=16, c_out=16, stride=1, ratio=ratio),
            GhostBlock(c_in=16, c_out=16, stride=1, ratio=ratio),
        )  # (N, 16, 32, 32)

        # Stage 2: downsample + kanal artırma
        self.stage2 = nn.Sequential(
            GhostBlock(c_in=16, c_out=32, stride=2, ratio=ratio),  # (N, 32, 16, 16)
            GhostBlock(c_in=32, c_out=32, stride=1, ratio=ratio),  # (N, 32, 16, 16)
        )

        # Stage 3: tekrar downsample + kanal artırma
        self.stage3 = nn.Sequential(
            GhostBlock(c_in=32, c_out=64, stride=2, ratio=ratio),  # (N, 64, 8, 8)
            GhostBlock(c_in=64, c_out=64, stride=1, ratio=ratio),  # (N, 64, 8, 8)
        )

        # Global Average Pool + FC
        self.gap = nn.AdaptiveAvgPool2d((1, 1))   # (N, 64, 1, 1)
        self.fc  = nn.Linear(64, num_classes)     # (N, 64) -> (N, num_classes)

    def forward(self, x):
        # x: (N, 3, 32, 32)
        x = self.stem(x)      # (N, 16, 32, 32)
        x = self.stage1(x)    # (N, 16, 32, 32)
        x = self.stage2(x)    # (N, 32, 16, 16)
        x = self.stage3(x)    # (N, 64, 8, 8)
        x = self.gap(x)       # (N, 64, 1, 1)
        x = x.view(x.size(0), -1)  # (N, 64)
        x = self.fc(x)        # (N, num_classes)
        return x


# -------------------------------------------------
# Küçük test
# -------------------------------------------------
if __name__ == "__main__":
    model = GhostNetSmall(num_classes=10, ratio=2)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("Çıktı şekli:", y.shape)              # Beklenen: (4, 10)
    print("Toplam parametre:", count_params(model))
