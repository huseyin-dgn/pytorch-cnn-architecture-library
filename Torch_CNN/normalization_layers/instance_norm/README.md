# Instance Normalization (InstanceNorm2d) — ConvINAct + Basit AutoEncoder

## Nedir?

**InstanceNorm2d (IN)**, her örnek (instance) için ayrı normalize yapar.

- Normalize eksenleri: **(H, W)** (ve kanal başına)
- Batch boyutundan bağımsızdır (BN gibi batch istatistiği kullanmaz)

Özet:

- **BN:** batch genelinde istatistik → küçük batch’te kararsız
- **GN:** kanal grupları üzerinden → stabil
- **IN:** _her sample’ı kendi içinde_ normalize eder → stil / kontrast etkisini güçlü biçimde bastırır

---

## Bu implementasyonda IN nasıl kullanılıyor?

### `ConvINAct` bloğu

Bu blok: **Conv → InstanceNorm2d → Activation**

Önemli ayarlar:

- `track_running_stats=False`  
  IN her zaman “anlık sample istatistiği” ile çalışır, running mean/var tutmaz.
- `affine=True/False`
  - `affine=True`: learnable gamma/beta var (daha esnek)
  - `affine=False`: sadece normalize eder (daha “sert”)

Activation seçenekleri:

- relu / silu / gelu / none

---

## SimpleIN_AutoEncoder neyi göstermek için iyi?

Bu model IN için “mantıklı” bir örnek:

### Neden?

IN çoğu zaman:

- **image-to-image** işlerinde (denoise, reconstruction, style transfer)
- **appearance / illumination** varyasyonunu azaltmak için
  iyi çalışır.

Autoencoder da tam olarak bu tipte bir yapıdır:

- encoder ile temsil sıkıştır
- decoder ile geri üret

---

## Modelin akışı (yüksek seviye)

### Encoder

- `enc1`: girişten base kanala + downsample (/2)
- `enc2`: kanal ikiye katlanır + downsample (/4)

### Bottleneck

- birkaç conv ile temsil güçlendirme

### Decoder

- `Upsample x2` + conv ile çözünürlük geri getirme
- en sonda 3 kanala projeksiyon
- `sigmoid` ile çıkışı [0,1] aralığına sıkıştırma

Bu “sigmoid” şunu ima eder:

> Çıkışı görüntü gibi düşünüp normalize edilmiş piksel aralığına çekiyorsun.

---

## IN ne zaman iyi fikir?

✅ Style transfer / domain invariant representation  
✅ Autoencoder / image reconstruction  
✅ Batch küçükse ve BN istemiyorsan  
✅ Görüntü kontrastı / ışık değişimlerinden etkilenmeyi azaltmak istiyorsan

❌ Saf classification backbone’unda bazen zarar verebilir  
Çünkü IN “stil bilgisi” ile beraber bazı discriminative sinyalleri de silebilir.

---

## Pratik notlar

- `in_affine=True` genelde daha iyi (modelin “normalize ettim ama geri ayarlayayım” esnekliği var)
- Upsample için nearest basit ve hızlı; daha kaliteli istersen conv transpose / bilinear + conv denenebilir
- IN + ReLU klasik; ama SiLU da daha yumuşak gradient verebilir

---

## Kısa Özet

Bu klasör:

- IN’i pratik bir blok (`ConvINAct`) olarak sunuyor
- IN’in güçlü olduğu “image-to-image” tipine uygun olarak basit bir AutoEncoder örneği veriyor
- `affine` kontrolü ile IN’in sertliğini/esnekliğini ayarlamana izin veriyor
