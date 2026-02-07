## RepVGG (Structural Re-parameterization)

RepVGG’nin olayı: **eğitimde çok dallı (multi-branch) yapı kullanıp, inference’ta hepsini tek bir 3×3 Conv’a katlamak.**  
Yani training’de optimizasyon kolay, deploy’da ise grafik sade ve hızlı. ⚡

### Eğitim modu (deploy=False) blok yapısı

`RepVGGBlock` içinde aynı input’a 3 yol var:

- **3×3 Conv + BN**
- **1×1 Conv + BN**
- **Identity + BN** (sadece `cin==cout` ve `stride==1` ise)

Çıkış:
`out = branch3x3(x) + branch1x1(x) + identity(x)`  
sonra `ReLU`

Bu multi-branch yapı ResNet gibi daha rahat öğrenir.

### Deploy modu (deploy=True) ne oluyor?

Deploy’da blok şuna dönüşüyor:

- **tek bir 3×3 Conv (bias=True)** + ReLU

Yani inference sırasında:

- branch yok
- BN yok
- toplama yok  
  → tek conv, tek aktivasyon ✅

### “Katlama” (re-parameterization) nasıl yapılıyor?

`switch_to_deploy()` çağrılınca:

1. Her branch’teki **Conv + BN** birleşiyor (fuse):

- `fuse_conv_bn` ile 3×3 ve 1×1 branch’lerin ağırlık+bias’ı çıkarılır

2. **1×1 kernel**, 3×3’e çevrilir:

- `pad_1x1_to_3x3`: 1×1 ağırlığı 3×3’ün ortasına koyar (diğerleri 0)

3. Identity (varsa) da “conv gibi” düşünülür:

- identity + BN → eşdeğer bir 3×3 kernel (merkezde 1’ler gibi) + bias üretilir

4. Hepsi toplanır:

- `W_eq = W3 + W1(padded) + W_id`
- `b_eq = b3 + b1 + b_id`

5. Yeni tek conv’a yazılır, eski branch’ler silinir.

### Bu modelde nerede kullanılıyor?

`RepVGGClassifier`:

- Stem normal conv
- Stage1/2/3: RepVGGBlock dizileri (stride=2 ile downsample)
- GAP + FC

`switch_to_deploy()` bütün block’ları tek conv’a çevirir.

### Testte ne kontrol ediliyor?

- Aynı input için:
  - training graph çıktısı (`y_train`)
  - deploy graph çıktısı (`y_deploy`)
- `max_abs_diff` küçükse re-param doğru çalışmış demektir ✅

### Dikkat

- `switch_to_deploy()` çağırmadan önce genelde `model.eval()` şart (BN istatistikleri sabitlenmiş olmalı).
- Bu kodda `fuse_conv_bn`, `pad_1x1_to_3x3`, `fuse_identity_bn` fonksiyonları dışarıdan gelmeli; onlar olmadan deploy dönüşümü çalışmaz.
