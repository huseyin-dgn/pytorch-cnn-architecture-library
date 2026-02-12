import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self,
                 max_drop: float = 0.0,
                 layer_idx: int | None = None,
                 num_layers: int | None = None,
                 warmup_steps: int = 0,
                 batchwise: bool = True,
                 store_mask: bool = True):
        super().__init__()

        self.max_drop = float(max_drop)
        # max_drop: float = 0.0 :: DropPath’in ulaşabileceği maksimum drop oranı..
        # 0.1 demek: her forward’da residual yolun %10 ihtimalle kapatılması.

        self.layer_idx = layer_idx
        # Bu DropPath’in bağlı olduğu bloğun derinlik indeksi.
        # Eğer None ise: katmana göre ölçekleme yapılmaz, direkt max_drop kullanılır.

        self.num_layers = num_layers
        # Anlamı: Schedule uygulayabilmek için toplam kaç blok/layer olduğunu söyler.
        # layer_idx ile beraber çalışır.
        # num_layers <= 1 veya None ise schedule devre dışı kalır

        self.warmup_steps = int(warmup_steps)
        # DropPath’in gücünü ilk N adımda yavaş yavaş artırmak.
        # Training başında modeli destabilize etmemek için kullanılır.

        self.batchwise = bool(batchwise)
        # Maskeyi nasıl uygulayacağın.
        #   True ise :: her örnek için: residual yol ya tamamen açık ya tamamen kapalı
        #   False ise :: element-wise dropout’a benzer (her piksel/eleman ayrı düşebilir)
        #       bu artık “DropPath” değil, daha çok Dropout benzeri davranır.

        self.store_mask = bool(store_mask)
        # Debug için son kullanılan maskeyi self.last_mask içine saklasın mı?
        #   True yaparsan:
        #       self.last_mask dolu olur (detach edilip saklanır)
        #       görselleştirme / debug kolaylaşır
        #   Normal eğitimde genelde False (ama sen True istedin)

        # step sayacı: model train moddayken her forward'da 1 artar (batch sayısı gibi düşün)
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.last_mask = None

    def layer_scaled_drop(self) -> float:
        # Eğer derinlik bazlı schedule bilgisi yoksa:
        #   - layer_idx None ise (hangi katmanda olduğumuzu bilmiyorsak)
        #   - num_layers None ise (toplam kaç katman var bilmiyorsak)
        #   - num_layers <= 1 ise (tek katmanlı bir yapıysa)
        #
        # Bu durumda derinliğe göre ölçekleme YAPILMAZ,
        # DropPath oranı sabit olarak max_drop kullanılır.
        if self.layer_idx is None or self.num_layers is None or self.num_layers <= 1:
            return self.max_drop

        # layer_idx: şu an hangi residual bloktayız (0-indexed)
        # num_layers - 1:
        #   çünkü layer index'leri 0'dan başlar
        #   en derin blok için:
        #       layer_idx = num_layers - 1
        #       frac = 1.0  (maksimum drop'a ulaşsın diye)
        #
        # frac: modelin DERİNLİĞİNDE ne kadar ilerlediğimizi gösteren oran
        frac = float(self.layer_idx) / float(self.num_layers - 1)

        # Gerçek uygulanacak DropPath oranı:
        # - sığ katmanlarda frac küçük → drop küçük
        # - derin katmanlarda frac büyük → drop büyük
        return self.max_drop * frac

    def warmup_drop_prob(self, drop: float) -> float:
        # drop: burada warmup uygulanmadan önceki HEDEF DropPath oranı.
        # warmup_steps: bu hedef orana KAÇ ADIMDA ulaşacağımız.
        if self.warmup_steps <= 0:
            return drop

        # s: o anki step (şu ana kadar kaç tane forward / eğitim adımı geçti?)
        s = int(self.step.item())

        # t: warmup ilerleme oranı (0..1)
        t = min(max(s / float(self.warmup_steps), 0.0), 1.0)

        # effective_drop = drop * t
        return drop * t

    def current_drop_prob(self) -> float:
        # “şu anda uygulanacak gerçek (efektif) drop oranı kaç?”
        d = self.layer_scaled_drop()
        d = self.warmup_drop_prob(d)

        # güvenlik clamp
        if d < 0.0:
            d = 0.0
        if d >= 1.0:
            d = 0.999
        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.step += 1

        drop_prob = self.current_drop_prob()

        if drop_prob <= 0.0 or (not self.training):
            if self.store_mask:
                self.last_mask = None
            return x

        keep_prob = 1.0 - drop_prob

        if self.batchwise:
            mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B,1,1,1...)
        else:
            mask_shape = x.shape  # element-wise

        mask = torch.empty(mask_shape, device=x.device, dtype=torch.float32).bernoulli_(keep_prob)
        mask = mask.to(dtype=x.dtype)

        if self.store_mask:
            self.last_mask = mask.detach()

        return x * mask / keep_prob


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, act="relu"):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act in ("silu", "swish"):
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError("act: 'relu' | 'silu'")

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BasicResBlockDP(nn.Module):
    def __init__(self, cin, cout, stride=1, act="relu", droppath: DropPath | None = None):
        super().__init__()
        self.conv1 = ConvBNAct(cin, cout, k=3, s=stride, p=1, act=act)
        self.conv2 = nn.Sequential(
            nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.SiLU(inplace=True)
        self.dp = droppath  # bu blok için DropPath objesi (oranı layer_idx/num_layers ile ayarlı)

        self.skip = nn.Identity()
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.conv2(out)

        # DropPath residual branch'e uygulanır (toplama öncesi)
        if self.dp is not None:
            out = self.dp(out)

        out = self.act(out + identity)
        return out


class ResNetSmallWithDropPath(nn.Module):
    def __init__(self, num_classes=100, act="relu", max_drop=0.1, warmup_steps=0, batchwise=True):
        super().__init__()

        # toplam residual blok sayısı: 6
        self.total_blocks = 6
        self._block_idx = 0

        self.stem = ConvBNAct(3, 64, k=3, s=1, p=1, act=act)

        # stage1 (32x32)
        self.stage1 = nn.Sequential(
            self._make_block(64, 64, stride=1, act=act, max_drop=max_drop, warmup_steps=warmup_steps, batchwise=batchwise),
            self._make_block(64, 64, stride=1, act=act, max_drop=max_drop, warmup_steps=warmup_steps, batchwise=batchwise),
        )

        # stage2 (16x16)
        self.stage2 = nn.Sequential(
            self._make_block(64, 128, stride=2, act=act, max_drop=max_drop, warmup_steps=warmup_steps, batchwise=batchwise),
            self._make_block(128, 128, stride=1, act=act, max_drop=max_drop, warmup_steps=warmup_steps, batchwise=batchwise),
        )

        # stage3 (8x8)
        self.stage3 = nn.Sequential(
            self._make_block(128, 256, stride=2, act=act, max_drop=max_drop, warmup_steps=warmup_steps, batchwise=batchwise),
            self._make_block(256, 256, stride=1, act=act, max_drop=max_drop, warmup_steps=warmup_steps, batchwise=batchwise),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def _make_block(self, cin, cout, stride, act, max_drop, warmup_steps, batchwise):
        # Her blok için ayrı DropPath instance oluşturuyoruz.
        # layer_idx arttıkça DropPath oranı artacak (schedule).
        dp = DropPath(
            max_drop=max_drop,
            layer_idx=self._block_idx,
            num_layers=self.total_blocks,
            warmup_steps=warmup_steps,
            batchwise=batchwise,
            store_mask=False,  
        )
        self._block_idx += 1

        return BasicResBlockDP(cin, cout, stride=stride, act=act, droppath=dp)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)

if __name__ == "__main__":
    model = ResNetSmallWithDropPath(num_classes=100, max_drop=0.2, warmup_steps=1000)
    x = torch.randn(4, 3, 32, 32)
    model.train()
    y = model(x)
    print("train:", y.shape)

    model.eval()
    y2 = model(x)
    print("eval:", y2.shape)