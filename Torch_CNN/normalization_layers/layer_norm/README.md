# Layer Normalization (LayerNorm)

## Nedir?

**LayerNorm (LN)**, normalizasyonu **batch boyutundan bağımsız** yapar.  
BatchNorm gibi “batch ortalaması/variance’ı” kullanmaz; her örnek için kendi içinde normalize eder.

Bu implementasyonda LN, CNN formatı olan **NCHW** için uyarlanmış:

> Her uzamsal noktada `(H, W)` **kanallar (C) boyunca** normalize edilir.

Yani her piksel konumunda kanal vektörünü normalize ediyorsun.

---

## `LayerNorm2d` tam olarak ne yapıyor?

### Giriş/Çıkış

- Giriş: `(N, C, H, W)`
- Çıkış: `(N, C, H, W)`

### Normalize ekseni

- `mean` ve `var` **C ekseninde** hesaplanır:

- `mean = mean(x over C)` → `(N, 1, H, W)`
- `var  = var(x over C)` → `(N, 1, H, W)`

Sonra standartlaştırma:

- `xhat = (x - mean) / sqrt(var + eps)`

Ve learnable affine:

- `gamma` (scale) ve `beta` (shift) kanal başına uygulanır.

---

## BN / GN / IN ile farkı ne?

| Yöntem      | Batch’e bağlı mı? | Normalize ekseni                 | Tipik kullanım                                 |
| ----------- | ----------------- | -------------------------------- | ---------------------------------------------- |
| **BN**      | ✅                | (N,H,W) üzerinden (kanal başına) | büyük batch + hızlı backbone                   |
| **GN**      | ❌                | kanal grupları                   | küçük batch detection                          |
| **IN**      | ❌                | (H,W) (kanal başına)             | style / image-to-image                         |
| **LN (bu)** | ❌                | **C** (her pikselde)             | transformer variasyonları, bazı modern CNN’ler |

Bu LN2d yaklaşımı özellikle:

- batch küçükse
- BN stabil değilse
- “her noktada kanal vektörünü” normalize etmek istiyorsan

mantıklı.

---

## `SimpleLN_CNN` neyi gösteriyor?

Bu model basit bir sınıflandırma backbone’u:

- Conv → LN2d → ReLU → Pool
- birkaç kez tekrar
- en sonda AdaptiveAvgPool + Linear

Amaç:

- LN2d’nin CNN içinde “drop-in norm layer” gibi çalıştığını göstermek.

---

## Ne zaman LN2d iyi fikir?

✅ Batch küçükse ve BN istemiyorsan  
✅ Training’de istatistik dalgalanması istemiyorsan  
✅ CNN’i “transformer benzeri norm” yaklaşımıyla stabilize etmek istiyorsan  
✅ GN’e alternatif arıyorsan

❌ Eğer klasik CNN performansı hedefleniyorsa ve batch büyükse BN daha hızlı/oturmuş olabilir  
❌ Çok derin ağlarda LN yerine GN/BN tercih eden pratikler daha yaygın olabilir

---

## Pratik notlar

- Bu LN2d implementasyonu “basit ve açık” bir versiyon.
- PyTorch’un `nn.LayerNorm`’u NCHW’te direkt kullanmak için genelde permute gerekir; burada gerek yok.
- `gamma/beta` kanal başına olduğu için LN’in affine kısmı “doğal” şekilde CNN’e uyuyor.

---

## Kısa Özet

`LayerNorm2d`:

- CNN’de `(N,C,H,W)` formatında
- her `(H,W)` noktasında **C boyunca** normalize eder
- batch’ten bağımsız stabil normalizasyon sağlar
- BN yerine kullanılabilecek bir alternatif norm katmanıdır
