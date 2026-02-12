import torch
import torch.nn as nn

class SpatialDropout2d(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()

        # p: drop olasılığı. 0 <= p < 1 olmalı (1 olursa her şeyi silersin, eğitim biter)
        if not (0.0 <= p < 1.0):
            raise ValueError("p 0 ve 1 arasında olmalı")

        # p'yi float'a çevirip saklıyoruz
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial Dropout sadece TRAIN modunda çalışır.
        # model.eval() çağrılırsa self.training False olur ve direkt x döner.
        # p=0 ise zaten hiçbir şey düşürmeyeceğiz.
        if (not self.training) or (self.p == 0.0):
            return x

        # q: keep probability (kanalı tutma olasılığı)
        # p=0.1 ise q=0.9 demek: kanalların %90'ı kalır, %10'u komple gider.
        q = 1.0 - self.p

        # mask boyutu: [B, C, 1, 1]
        # - B: batch
        # - C: kanal sayısı
        # - 1,1: H ve W boyunca aynı maskeyi kullanacağız (broadcast)
        #
        # bernoulli_(q):
        # - her kanal için 1 gelirse kanal tutulur
        # - 0 gelirse kanal komple sıfırlanır
        mask = torch.empty(
            (x.size(0), x.size(1), 1, 1),      # [B, C, 1, 1]
            device=x.device,                   # mask cihazı x ile aynı (cpu/gpu)
            dtype=x.dtype                      # mask dtype'ı x ile aynı
        ).bernoulli_(q)

        # x * mask:
        # - mask=0 olan kanallar komple 0 olur
        # - mask=1 olan kanallar aynen kalır
        #
        # / q (inverted dropout scaling):
        # - beklenen aktivasyon seviyesi korunur
        # - yani train'de dropout yapınca magnitude düşmesin diye ölçekliyoruz
        return x * mask / q