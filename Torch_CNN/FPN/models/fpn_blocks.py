import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNBlock(nn.Module):
    """
    Ortalama (klasik) FPN:
      - Lateral 1x1 conv (kanal hizalama)
      - Top-down upsample + element-wise sum
      - 3x3 smooth conv
    Girdi:  [C2, C3, C4, C5]
    Çıktı:  [P2, P3, P4, P5]
    """
    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256, upsample_mode="nearest"):
        super().__init__()
        self.upsample_mode = upsample_mode

        # Ck -> out_channels
            ## Toplayabilmek için hepsini aynı kanal sayısına getiriyoruz:
            ## Lk = Conv1x1(Ck) → hepsi (B, out_channels, Hk, Wk)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(cin, out_channels, kernel_size=1, bias=False)
            for cin in in_channels
        ])

        # Pk smoothing
            ## Bu karışımı daha lokal bağlamla düzeltmek.
            ## 3×3 mekansal bağlamı işler.
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            for _ in in_channels
        ])

    def forward(self, feats):
        C2, C3, C4, C5 = feats # İçindeki 4 tensörü sırasıyla C2, C3, C4, C5 isimlerine atıyor

        # 1) lateral projections
        L2 = self.lateral_convs[0](C2) # L2 = C2’nin 1×1’den geçmiş hali
        L3 = self.lateral_convs[1](C3)
        L4 = self.lateral_convs[2](C4)
        L5 = self.lateral_convs[3](C5)

        # 2) top-down + sum
            ## interpolate :: :: Bir feature map’i farklı bir uzaysal boyuta büyütmek veya küçültmek.
            ## upsample_mode :: :: parametresi büyütme yöntemini belirler. 
                ### nearest :: => Hızlı.Bloklu olabilir.Detection’da sık kullanılır
                ### bilinear :: => 4 komşu pikselden ağırlıklı ortalama.Daha yumuşak.Segmentation’da sık kullanılır
        P5 = L5
        P4 = L4 + F.interpolate(P5, size=L4.shape[-2:], mode=self.upsample_mode)
        P3 = L3 + F.interpolate(P4, size=L3.shape[-2:], mode=self.upsample_mode)
        P2 = L2 + F.interpolate(P3, size=L2.shape[-2:], mode=self.upsample_mode)
        ## Yukarıdaki kodun anlamı => P5 i al L4 ün H,W boyutuna getir.Nearest yöntemi kullan

        # 3) smooth
        P2 = self.smooth_convs[0](P2)
        P3 = self.smooth_convs[1](P3)
        P4 = self.smooth_convs[2](P4)
        P5 = self.smooth_convs[3](P5)

        return [P2, P3, P4, P5]


# quick test
if __name__ == "__main__":
    B = 2
    C2 = torch.randn(B, 256, 128, 128)
    C3 = torch.randn(B, 512, 64, 64)
    C4 = torch.randn(B, 1024, 32, 32)
    C5 = torch.randn(B, 2048, 16, 16)

    fpn = FPNBlock(in_channels=(256, 512, 1024, 2048), out_channels=256)
    P2, P3, P4, P5 = fpn([C2, C3, C4, C5])
    print([p.shape for p in (P2, P3, P4, P5)])
