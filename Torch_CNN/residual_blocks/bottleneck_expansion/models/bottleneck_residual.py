import torch
import torch.nn as nn

# ============================================================
# Bottleneck Residual Block (ResNet-50/101 tarzı)
# Mantık:
#   F(x) = 1x1 (daralt) -> 3x3 (asıl iş) -> 1x1 (genişlet)
#   out = ReLU( F(x) + skip(x) )
#
# Neden var?
# - Derin ağlarda gradient akışını korur (skip path).
# - 3x3'ü daha az kanalda çalıştırıp hesap maliyetini düşürür (bottleneck).
# - "Expansion" (genelde 4) ile çıkış kanalı büyür: out_ch = mid_ch * expansion
#
# Nerede kullanılır?
# - ResNet-50/101/152 backbone stage'lerinde (özellikle 256/512/1024/2048 kanallar).
# - Detection backbone’larında (FPN öncesi C2/C3/C4/C5 stage'leri).
# ============================================================

class BottleneckResidual(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expansion: int = 4,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU,
    ):
        """
        in_ch   : giriş kanal sayısı
        out_ch  : blok çıkış kanal sayısı (genelde stage'in kanal sayısı)
        stride  : 1 ise çözünürlük aynı, 2 ise downsample (H,W yarıya iner)
        expansion: bottleneck oranı (ResNet'te tipik 4)
                  mid_ch = out_ch // expansion
        """
        super().__init__()

        if out_ch % expansion != 0:
            raise ValueError("out_ch, expansion'a bölünebilir olmalı (ör: 256 ve expansion=4).")

        mid_ch = out_ch // expansion  # bottleneck'in dar kanal sayısı

        # ----------------------------
        # Main path: F(x)
        # ----------------------------

        # (1) 1x1 "reduce": kanalı daraltır
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1   = norm_layer(mid_ch)

        # (2) 3x3 "compute": asıl feature çıkarımı burada
        # stride burada uygulanır -> downsample gerekiyorsa bu conv stride=2 olur
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = norm_layer(mid_ch)

        # (3) 1x1 "expand": kanalı tekrar büyütür (out_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3   = norm_layer(out_ch)

        # Aktivasyon (ResNet v1 post-act mantığı: toplama sonrası ReLU)
        self.act = act_layer(inplace=True)

        # ----------------------------
        # Skip path: identity veya projection
        # ----------------------------
        # Eğer:
        # - stride=2 ise (çözünürlük değişiyor) veya
        # - in_ch != out_ch ise (kanal sayısı değişiyor)
        # o zaman skip'i de aynı shape'e getirmek zorundayız: 1x1 projection
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False),
                norm_layer(out_ch),
            )

    def forward(self, x):
        # Skip branch
        skip = x if self.proj is None else self.proj(x)

        # Main branch: F(x)
        y = self.act(self.bn1(self.conv1(x)))      # 1x1 reduce
        y = self.act(self.bn2(self.conv2(y)))      # 3x3 compute
        y = self.bn3(self.conv3(y))                # 1x1 expand (aktivasyon yok burada)

        # Residual sum
        out = self.act(skip + y)                   # toplama + ReLU (post-act)
        return out


# ============================================================
# Bottleneck Residual'ın "model içinde kullanımı" (örnek stage)
# ============================================================

class TinyResNetStage(nn.Module):
    """
    Örnek: Bir stage düşün:
      giriş: 64 kanal
      stage çıkışı: 256 kanal (expansion=4 -> mid=64)
    İlk blok stride=2 ile downsample yapabilir (C2->C3 gibi)
    """
    def __init__(self):
        super().__init__()

        self.block1 = BottleneckResidual(in_ch=64, out_ch=256, stride=2, expansion=4)
        self.block2 = BottleneckResidual(in_ch=256, out_ch=256, stride=1, expansion=4)
        self.block3 = BottleneckResidual(in_ch=256, out_ch=256, stride=1, expansion=4)

    def forward(self, x):
        x = self.block1(x)   # burada çözünürlük yarıya iner, kanal 256 olur
        x = self.block2(x)
        x = self.block3(x)
        return x


# ============================================================
# Hızlı test: shape kontrolü
# ============================================================
if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)    # batch=2, 64 kanal, 32x32
    stage = TinyResNetStage()
    y = stage(x)
    print("input :", x.shape)
    print("output:", y.shape)          # beklenen: (2, 256, 16, 16)
