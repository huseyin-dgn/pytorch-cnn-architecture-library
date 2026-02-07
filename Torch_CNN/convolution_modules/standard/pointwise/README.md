## Pointwise Conv (1×1)

Pointwise conv = **1×1 konvolüsyon**.  
Uzamsal olarak bir şey “görmez” (komşuya bakmaz). Aynı piksel konumunda sadece **kanalları karıştırır** ve kanal sayısını değiştirir. 🎛️

### 1×1 ne işe yarar?

- **Channel mixing:** `(C_in -> C_out)` dönüşümü yapar
- **Bottleneck / expand:** kanalı düşürüp yükseltebilirsin
- **Compute kontrolü:** 3×3’ün yanına koyup mimariyi daha verimli kurarsın

### Bu kodda blok mantığı

`ConvPointwiseBlock` sırası:

1. **3×3 Conv (spatial feature)**
   - kenar, doku, lokal bilgi çıkarır
   - `stride` burada varsa downsample eder

2. **1×1 Pointwise Conv (channel mixing)**
   - 3×3’ten çıkan feature’ları kanallar arasında karıştırır
   - kanal sayısını `c_out`’a ayarlar

Her adımın arkasında `BN + ReLU` var → stabil ve hızlı öğrenir.

### Model akışı (`PointwiseNetSmall`)

- Stem: 3→16
- Stage1: 16→32 (boyut sabit)
- Stage2: stride=2 ile 32→64 (32→16 spatial)
- Stage3: stride=2 ile 64→128 (16→8 spatial)
- GAP + FC

Yani 3×3 ile “göz” var, 1×1 ile “kanal beynini” karıştırıyorsun.

### Dikkat

- 1×1 tek başına spatial bilgi taşımaz; **3×3 / shift / depthwise** gibi bir spatial işlemle eşleşince anlamlı olur.
- Bu tasarım, klasik “Conv-BN-ReLU + 1×1” şablonunun sade hali.
