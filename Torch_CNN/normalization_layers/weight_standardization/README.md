# Weight Standardization (WS

## Nedir?

**Weight Standardization (WS)**, Convolution ağırlıklarını (kernel weight) her forward’da **standartlaştırarak** (mean=0, std=1) eğitim stabilitesini artırmayı hedefler.

Önemli nokta:

> WS, **aktivasyonları değil**, **ağırlıkları** normalize eder.

Bu, özellikle **GroupNorm (GN)** ile birlikte çok kullanılan bir kombinasyondur:

- batch bağımsız stabilite (GN)
- ağırlık ölçeğini sabitleme (WS)

---

## WSConv2d ne yapıyor?

Bu implementasyonda WS, `nn.Conv2d`’den türetilmiş bir katmanda uygulanıyor.

### Adımlar

1. Ham ağırlığı al:

- `w = self.weight`

2. Her `out_channel` için mean çıkar:

- `w = w - mean(w)` _(kernel merkezleme)_

3. Her `out_channel` için std’ye böl:

- `w = w / (std(w) + eps)` _(ölçek sabitleme)_

4. Standartlaştırılmış `w` ile convolution yap:

- `F.conv2d(..., w, ...)`

### Neden out_channel bazında?

Çünkü Conv2d’de her çıkış kanalı kendi kernel setiyle üretilir.  
WS bu kernel setini “birim ölçek”e çekerek gradient akışını daha stabil hale getirir.

---

## Neden WS + GN birlikte kullanılıyor?

BN “istatistik” üzerinden stabilite getirir ama batch’e bağımlıdır.  
GN batch bağımsızdır fakat ağırlık ölçeği drift edebilir.

Bu yüzden modern pratikte sık görülen eşleşme:

> **WSConv2d + GroupNorm**

WS:

- weight dağılımını her forward’da sabitler

GN:

- activations’ı batch bağımsız stabilize eder

Sonuç:

- küçük batch’te stabil ve tutarlı eğitim

---

## Bu klasördeki bloklar

### 1) `WSGNAct`

**WSConv2d → GroupNorm → SiLU**

- WS burada “conv” katmanına uygulanır
- Group sayısı, `out_ch` bölünecek şekilde geriye doğru düşürülerek ayarlanır

### 2) `WSResBlock`

Residual blok:

- 2 adet WSGNAct
- gerekirse projection (1x1 WSConv + GN)
- residual toplama sonrası SiLU

Amaç:

- WS+GN kombinasyonunun residual network içinde nasıl durduğunu göstermek

---

## WSResUNet: WS+GN tabanlı U-Net

Bu dosyada WS, sadece bir “layer trick” değil; komple U-Net backbone’unda tutarlı şekilde kullanılıyor.

### Encoder

- stem (7x7 WSConv + GN)
- stage1..stage4 (WSResBlock)
- çözünürlük /2, /4, /8, /16, /32 şeklinde iner

### Bridge

- bottleneck (iki WSGNAct)

### Decoder

- bilinear upsample + reduce + skip concat + refine
- her yerde WSGNAct kullanılır

### Head

- 1x1 conv ile `num_classes` kanalına projeksiyon

Bu yapı, segmentation gibi işlerde WS+GN’in “gerçek” kullanımını gösterir.

---

## Ne zaman WS mantıklı?

✅ Batch küçükse (segmentation/detection eğitimleri gibi)  
✅ GN/LN gibi batch bağımsız normlar kullanıyorsan  
✅ Training kararsızsa, gradient patlaması/ölçek drift görüyorsan  
✅ ResNet/U-Net gibi derin bloklarda stabilite istiyorsan

❌ Büyük batch + BN ile zaten stabil bir setup varsa (fayda daha az olabilir)  
❌ Inference optimizasyonu BN-fusion üzerine kuruluyken (WS forward maliyeti getirebilir)

---

## Pratik ipuçları

- WS + GN kombinasyonu özellikle küçük batch’te “BN alternatifi” gibi düşünülür.
- `groups=32` klasik başlangıçtır ama kanal sayısına bölünmüyorsa grup sayısı düşürülür (bu dosyada var).
- SiLU aktivasyonu, ReLU’ya göre daha yumuşak gradient sağlayabilir.

---

## Kısa Özet

Bu modül:

- Conv ağırlıklarını forward’da standartlaştıran **WSConv2d** içerir
- WS’yi **GroupNorm** ile eşleştirerek batch bağımsız stabil eğitim hedefler
- U-Net benzeri bir mimari üzerinde WS+GN kullanımını uçtan uca örnekler
