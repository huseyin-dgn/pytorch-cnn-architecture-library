import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """
    Basit bir "Conv -> (BN) -> (Activation)" bloğu.

    Neden var?
    - FPN içindeki lateral (1x1) ve smooth (3x3) conv'ları tekrar tekrar yazmamak için.
    - İstersen BN ve aktivasyon ekleyip stabiliteyi artırırsın (özellikle scratch training'de).
    """
    def __init__(self, cin, cout, k=1, s=1, p=0, bn=True, act=None):
        super().__init__()

        # Conv2d:
        # - cin: giriş kanal sayısı
        # - cout: çıkış kanal sayısı (FPN'de hepsini aynı "out_channels" yapıyoruz)
        # - k: kernel size (1: kanal hizalama, 3: spatial smoothing)
        # - s: stride (FPN'de genelde 1 kalır)
        # - p: padding (k=3 için p=1 ile HxW korunur)
        #
        # bias=not bn:
        # - BN kullanıyorsan (bn=True), bias çoğu zaman gereksizdir (BN offset öğrenir).
        # - BN yoksa (bn=False), conv bias'ı açık bırakmak mantıklı.
        self.conv = nn.Conv2d(cin, cout, k, s, p, bias=not bn)

        # BatchNorm2d:
        # - Aktivasyonların dağılımını stabilize eder.
        # - Küçük batch'te bazen sıkıntı olabilir; bu yüzden opsiyonel.
        self.bn = nn.BatchNorm2d(cout) if bn else nn.Identity()

        # Aktivasyon:
        # - SiLU genelde modern CNN'lerde iyi çalışır.
        # - ReLU daha klasik ve hızlı.
        # - None ise aktivasyon yok (Identity).
        if act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        # Sıra: Conv -> BN -> Act
        return self.act(self.bn(self.conv(x)))


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) - klasik top-down yapı.

    Girdi:
    - feats = [C2, C3, C4, C5]
      (Backbone'un farklı çözünürlüklerde ürettiği feature map'ler)

    Çıktı:
    - [P2, P3, P4, P5]
      (Her seviyede aynı kanal sayısına (out_channels) normalize edilmiş piramit feature'lar)
    """
    def __init__(self,
                 in_channels_list,          # [C2_ch, C3_ch, C4_ch, C5_ch]
                 out_channels=256,          # P2-P5 hepsinin kanal sayısı
                 upsample_mode="nearest",   # "nearest" veya "bilinear"
                 use_bn=False,              # lateral/smooth conv sonrası BN kullan
                 use_act=False):            # lateral/smooth conv sonrası aktivasyon kullan
        super().__init__()

        # Upsample modunu saklıyoruz (top-down pathway'de kullanılacak)
        self.upsample_mode = upsample_mode

        # Aktivasyonu seç:
        # - use_act=True ise SiLU, değilse None (Identity)
        act = "silu" if use_act else None

        # LATERAL CONVS (1x1):
        # Amaç: backbone'dan gelen C2-C5 feature'larını aynı kanal boyutuna getirmek.
        #
        # Örn:
        # - C5: 2048 kanal olabilir
        # - C2: 256 kanal olabilir
        # FPN'de bunları out_channels=256 gibi ortak boyuta "projeksiyon" yaparız.
        self.lateral = nn.ModuleList([
            ConvBNAct(cin, out_channels, k=1, p=0, bn=use_bn, act=act)
            for cin in in_channels_list
        ])

        # SMOOTH CONVS (3x3):
        # Amaç: Top-down upsample + add sonrası oluşan "aliasing / checkerboard / karışık"
        # birleşimleri yumuşatmak ve lokal bağlamı toparlamak.
        #
        # 3x3, stride=1, padding=1 => HxW korunur.
        self.smooth = nn.ModuleList([
            ConvBNAct(out_channels, out_channels, k=3, p=1, bn=use_bn, act=act)
            for _ in in_channels_list
        ])

    def _upsample(self, x, ref):
        """
        x'i, ref feature map'in HxW boyutuna büyütür.

        Neden "size=ref.shape[-2:]"?
        - Scale factor kullanırsan bazı backbone'larda rounding yüzünden 1px mismatch çıkabiliyor.
        - Size ile direkt hedef HxW'ye kilitlersin.

        Bilinear seçeneği:
        - "bilinear" daha yumuşak upsample verir.
        - align_corners=False genelde segmentation/detection için güvenli varsayımdır.
        """
        if self.upsample_mode == "bilinear":
            return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

        # nearest:
        # - daha hızlı, YOLO tarzı head'lerde sık kullanılır
        return F.interpolate(x, size=ref.shape[-2:], mode=self.upsample_mode)

    def forward(self, feats):
        # Bu FPN, backbone'dan tam olarak 4 seviye bekliyor.
        assert len(feats) == 4, "FPN expects [C2,C3,C4,C5]"
        C2, C3, C4, C5 = feats

        # 1) LATERAL PROJEKSİYON (C -> L):
        # C2..C5 farklı kanal sayılarında gelir; hepsini out_channels'a indirgeriz.
        #
        # L2 = 1x1(C2), L3 = 1x1(C3), ...
        L2 = self.lateral[0](C2)
        L3 = self.lateral[1](C3)
        L4 = self.lateral[2](C4)
        L5 = self.lateral[3](C5)

        # 2) TOP-DOWN PATHWAY (P oluşturma):
        # En üst seviyeden başlarız.
        #
        # P5 = L5 (en kaba çözünürlük, en semantik seviye)
        P5 = L5

        # P4 = L4 + upsample(P5)
        # - P5'i L4'ün çözünürlüğüne çıkarırız
        # - sonra L4 ile toplayarak hem semantik (P5) hem detay (L4) harmanlarız
        P4 = L4 + self._upsample(P5, L4)

        # P3 = L3 + upsample(P4)
        P3 = L3 + self._upsample(P4, L3)

        # P2 = L2 + upsample(P3)
        P2 = L2 + self._upsample(P3, L2)

        # 3) SMOOTHING (3x3):
        # Top-down birleşim sonrası feature'ları 3x3 ile "temizleriz".
        # Bu conv'lar spatial bağlamı toparlar ve artefact'ları azaltır.
        P2 = self.smooth[0](P2)
        P3 = self.smooth[1](P3)
        P4 = self.smooth[2](P4)
        P5 = self.smooth[3](P5)

        # Çıktı sırası: küçükten büyüğe (yüksek çözünürlükten düşüğe)
        # P2: en yüksek çözünürlük (küçük objelerde değerli)
        # P5: en düşük çözünürlük (büyük objelerde / semantik)
        return [P2, P3, P4, P5]