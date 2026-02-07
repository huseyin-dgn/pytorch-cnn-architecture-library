# Coordinate Attention (CA)

## Amaç

Coordinate Attention (CA), klasik channel attention yaklaşımlarının aksine **uzamsal konum bilgisini tamamen kaybetmeden** kanal ölçekleme yapmayı hedefler.

Standart global pooling tabanlı attention’larda (SE/ECA gibi) tüm uzamsal bilgi tek vektöre sıkışır.  
CA ise şu fikri uygular:

> “Kanal önemini öğren, ama bunu konum (H/W) bilgisiyle birlikte yap.”

---

## Temel Fikir

CA, feature map üzerinde iki ayrı eksende profil çıkarır:

- **H yönü profili**: W üzerinde ortalama alarak (H bilgisini korur)
- **W yönü profili**: H üzerinde ortalama alarak (W bilgisini korur)

Bu iki profil birleştirilir, bottleneck üzerinden geçirilir ve iki ayrı attention maskesi üretilir:

- `a_h`: (B, C, H, 1) → yükseklik yönünde maske
- `a_w`: (B, C, 1, W) → genişlik yönünde maske

Sonra bu iki maske broadcast ile çarpılarak (B, C, H, W) ölçek elde edilir.

---

## Akış (Adım Adım)

1. **Yönsel Sıkıştırma (Directional Pooling)**
   - `x_h`: W ekseninde ortalama → (B,C,H,1)
   - `x_w`: H ekseninde ortalama → (B,C,W,1) (concat için permute edilir)

2. **Birleştirme**
   - `x_h` ve `x_w` birleştirilir → (B,C,H+W,1)

3. **Bottleneck (1×1 Conv + BN + Aktivasyon)**
   - Kanal bilgisi sıkıştırılır ve etkileşim öğrenilir:
   - Çıkış: (B, mid, H+W, 1)

4. **Ayrıştırma**
   - `y_h`: (B, mid, H, 1)
   - `y_w`: (B, mid, 1, W)

5. **Mask Üretimi**
   - `a_h`: sigmoid ile (B,C,H,1)
   - `a_w`: sigmoid ile (B,C,1,W)

6. **Birleşik Attention**
   - `att = a_h * a_w` → (B,C,H,W)

7. **Alpha ile Güç Kontrolü**
   - `alpha = 0` → attention etkisi kapalı (yaklaşık identity)
   - `alpha = 1` → tam attention
   - alpha ister sabit ister öğrenilebilir olabilir

---

## Parametreler

| Parametre          | Açıklama                                       |
| ------------------ | ---------------------------------------------- |
| `in_channels`      | Giriş kanal sayısı (C)                         |
| `reduction`        | Bottleneck oranı (C // reduction)              |
| `min_mid_channels` | Bottleneck alt sınırı (çok küçülmeyi engeller) |
| `act`              | Aktivasyon (`hswish` veya `relu`)              |
| `alpha`            | Attention gücü (0..1 önerilir)                 |
| `learnable_alpha`  | `True` ise alpha öğrenilir                     |

---

## Ne Tür Attention?

| Tür                                | Durum           |
| ---------------------------------- | --------------- |
| Channel Attention                  | ✔               |
| Spatial Attention (klasik 2D mask) | ⚠ Dolaylı       |
| Directional / Coordinate Awareness | ✔ (H/W yönünde) |
| Hybrid özellik                     | ✔               |

CA, klasik “spatial attention” gibi serbest bir 2D mask üretmez; bunun yerine **yönsel (H ve W) maske** üretir. Bu yüzden konum bilgisi daha kontrollü taşınır.

---

## Avantajlar

✔ Konum bilgisini koruyarak kanal ölçekleme  
✔ Hafif (1×1 conv + basit pooling)  
✔ Özellikle detection/backbone tarafında işe yarayabilir

---

## Dezavantajlar

✘ Tam serbest uzamsal (2D) dikkat üretmez  
✘ Çok küçük feature map’lerde (H/W çok küçükken) etkisi sınırlı olabilir

---

## İzleme (Monitoring)

Bu implementasyon, son üretilen:

- `a_h` ve `a_w` maskelerini saklar

`last_mask_stats()` ile min/mean/max/std gibi istatistikler alınarak attention davranışı izlenebilir.

---

## Kısa Özet

Coordinate Attention:

1. H ve W yönünde ayrı ayrı global profil çıkarır
2. Bottleneck ile kanal etkileşimi öğrenir
3. H ve W maskelerini üretir
4. Bu maskeleri birleştirip feature map’i ölçekler
5. `alpha` ile attention gücünü kontrol eder

> “Konum bilgisini kaybetmeden kanal dikkatini uygulayan yönsel attention.”
