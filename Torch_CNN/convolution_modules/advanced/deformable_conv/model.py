import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d  # DCN implementasyonu (torchvision içinde)


class DeformableConvBlock(nn.Module):
    """
    Bu blokta:
      - offset üreten bir Conv2d var
      - asıl feature'ı işleyen DeformConv2d var
      - ardından BN + ReLU geliyor

    Klasik ConvBlock gibi düşünebilirsin, tek fark:
      Conv2d yerine DeformConv2d + offset_conv kullanıyoruz.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 1) Offset üreten konvolüsyon
        # Her grid noktası için (dx, dy) → 2 * K * K kanal
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,  # offset için bias genelde açık bırakılır
        )

        # 2) Asıl deformable conv
        self.dcn = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        # 3) Normalizasyon + aktivasyon
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: (B, C_in, H, W)
        """
        # 1) Offset haritasını üret
        offset = self.offset_conv(x)
        # offset shape: (B, 2*K*K, H_out, W_out)

        # 2) Deformable Conv uygula
        out = self.dcn(x, offset)
        # out: (B, C_out, H_out, W_out)

        # 3) BN + ReLU
        out = self.bn(out)
        out = self.act(out)
        return out


class SmallDeformableCNN(nn.Module):
    """
    Örnek bir küçük CNN:
      - İlk katman: normal Conv2d
      - İkinci katman: DeformableConvBlock (DCN kullanıyor)
      - Sonunda global average pooling + Linear (classification head)

    Bunu kendi projende backbone iskeleti gibi düşünebilirsin.
    """
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        # 1) Normal konvolüsyon bloğu
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 2) Deformable Convolution bloğu (esas olay burada)
        self.deform_block = DeformableConvBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # 3) Bir conv daha (istersen yine DCN ile değiştirebilirsin)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 4) Sınıflandırma için head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        x: (B, in_channels, H, W)
        """
        # (B, 3, H, W) → (B, 32, H, W)
        x = self.conv1(x)

        # (B, 32, H, W) → (B, 64, H, W)  [DEFORMABLE CONV KULLANILAN YER]
        x = self.deform_block(x)

        # (B, 64, H, W) → (B, 128, H/2, W/2)
        x = self.conv3(x)

        # Global average pooling
        x = self.avgpool(x)            # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)      # (B, 128)

        # Sınıflandırma
        logits = self.fc(x)            # (B, num_classes)
        return logits


if __name__ == "__main__":
    # Dummy input ile test
    model = SmallDeformableCNN(in_channels=3, num_classes=10)
    dummy = torch.randn(4, 3, 64, 64)   # B=4, 64x64 RGB görüntü
    out = model(dummy)
    print("Input shape :", dummy.shape)
    print("Output shape:", out.shape)   # (4, 10) olmalı
