import torch
import torch.nn as nn

class StochasticDepth(nn.Module):
    """
    Stochastic Depth (DropPath'in CNN/Residual özel hali)

    Ne yapar?
    - SADECE training modunda çalışır.
    - Residual branch çıktısını (F(x)) bazı örneklerde tamamen 0'lar.
    - Bazı örneklerde ise branch'i tutar ve /q ile ölçekler (inverted scaling).

    ⚠️ ÖNEMLİ:
    Bu modülü genelde direkt feature map'e değil,
    residual bloğun branch çıktısına uygularsın:
        out = sd(F(x))
        y = x + out
    """
    def __init__(self, p: float = 0.1):
        super().__init__()

        # p: drop probability (branch'i düşürme olasılığı)
        # 0 <= p < 1 olmalı. p=1 olursa branch her zaman 0 olur.
        if not (0.0 <= p < 1.0):
            raise ValueError("p değeri 0 ve 1 arasında olmalı")

        # p'yi float'a çevirip saklıyoruz
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Eval modunda SD kapalıdır.
        # model.eval() -> self.training False olur -> x'i olduğu gibi döndürür.
        #
        # 2) p=0 ise zaten hiçbir şey düşürmeyeceğiz.
        if (not self.training) or self.p == 0.0:
            return x

        # q: keep probability (branch'i tutma olasılığı)
        # p=0.2 ise q=0.8 -> örneklerin %80'inde branch açık, %20'sinde kapalı
        q = 1.0 - self.p

        # mask shape (asıl fark burada):
        # x eğer [B, C, H, W] ise x.ndim = 4
        # shape = (B,) + (1,1,1) = (B, 1, 1, 1)
        #
        # Bu ne demek?
        # - Her örnek için TEK bir 0/1 karar veriyoruz (sample-wise)
        # - O karar tüm kanallar + tüm H,W boyunca aynı uygulanıyor
        # -> Yani bir örnekte branch ya tamamen var ya tamamen yok.
        shape = (x.size(0),) + (1,) * (x.ndim - 1)   # (B,1,1,1...) broadcast için

        # Bernoulli(q):
        # mask=1 -> branch tutulur
        # mask=0 -> branch komple 0 olur (o örnek için)
        mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(q)

        # inverted scaling (/q):
        # branch tutulduğu durumda büyüklüğü /q ile ölçekleriz.
        # Amaç: E[mask/q] = 1 olacak şekilde beklenen değeri korumak.
        #
        # Örnek:
        # q=0.8 ise mask 1 geldiğinde /0.8 ile büyütülür.
        return x * mask / q

def sd_linear_schedule(total_blocks: int, p_max: float):
    """
    p_i = p_max * i/(total_blocks-1)
    0'dan başlar, en derin blokta p_max olur.
    """
    if total_blocks <= 1:
        return [p_max]
    return [p_max * i / (total_blocks - 1) for i in range(total_blocks)]


class BasicResBlockSD(nn.Module):
    def __init__(self, cin, cout, stride=1, sd_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

        self.sd = StochasticDepth(p=sd_p)

        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.sd(out)                 # ✅ branch üzerinde
        out = self.act(out + identity)
        return out