# Coordinate Attention Plus (CA+)

## Amaç

**CoordinateAttPlus**, klasik _Coordinate Attention (CA)_ fikrini daha güçlü ve daha stabil hale getiren geliştirilmiş bir varyasyondur.

Klasik CA’nın ana hedefi:

- Kanal seçimi yaparken **konum/yön bilgisini (H ve W)** korumak

CA+ ise buna ek olarak:

- Daha zengin yönsel özellik çıkarımı (multi-scale)
- Daha kontrollü attention gücü
- İsteğe bağlı ek bir spatial gate
- İstatistik izleme (monitoring)

sunmayı amaçlar.

---

## Temel Fikir

CA ailesinde attention şu şekilde üretilir:

1. Feature map’ten iki ayrı yön profili çıkarılır

- **H profili:** W üzerinden özet (H bilgisini korur)
- **W profili:** H üzerinden özet (W bilgisini korur)

2. Bu profiller bir “ortak bottleneck” içinde işlenir
3. İki ayrı maske üretilir:

- `attn_h` → (B, C, H, 1)
- `attn_w` → (B, C, 1, W)

4. Bu maskeler çarpılarak (broadcast) ölçek elde edilir.

---

## CA+ Neyi Farklı Yapıyor?

### 1) Ortalama + Maksimum Profil Birleştirme

Klasik CA genelde sadece ortalama pooling kullanır.  
Bu versiyon her eksende:

- **mean**
- **max**

birleşimi ile profil çıkarır:

> Böylece hem “genel aktivasyon” hem “tepe aktivasyon” bilgisi taşınır.

---

### 2) Multi-Scale Yönsel Filtreleme (Local + Dilated)

Her eksen için iki farklı depthwise filtre uygulanır:

- **Local (3×1 / 1×3)** → kısa menzil
- **Dilated (3×1 / 1×3)** → daha geniş bağlam

Bu ikisi toplanıp “channel mixer” ile kanal etkileşimi güçlendirilir.

**Etki:**  
Farklı ölçeklerdeki yönsel örüntüler daha iyi yakalanır.

---

### 3) Ortak Bottleneck + Refine (2 aşamalı)

CA+ bottleneck’i tek adım değil, iki aşamalı kullanır:

- `proj` → sıkıştırma ve ortak temsil
- `refine` → bu temsilin rafine edilmesi

Arada norm + aktivasyon bulunur.

**Etki:**  
Mask üretimi daha kararlı ve daha ifade gücü yüksek olur.

---

### 4) Öğrenilebilir Alpha ile Güç Kontrolü (H ve W ayrı)

Bu yapı `attn_h` ve `attn_w` için ayrı ayrı bir “karışım katsayısı” öğrenir:

- `alpha_h`
- `alpha_w`

Maske doğrudan uygulanmaz; önce “identity ile karıştırılır”:

- alpha ↓ → maske etkisi azalır
- alpha ↑ → maske daha baskın olur

**Etki:**  
Training sırasında attention’ın aşırı baskılaması azalır.

---

### 5) Beta ile Ölçeklemeyi Yumuşatma

Maskeler birleştirildikten sonra ölçek, 1 etrafında yumuşatılır:

- beta = 0 → ölçek neredeyse identity davranır
- beta ↑ → attention etkisi artar

**Etki:**  
Attention’ın agresifliği daha kontrollü olur.

---

### 6) Opsiyonel Spatial Gate (Ek Kapı)

İstenirse CA+ çıktısına ek bir spatial gate uygulanır.

Bu, attention sonucunu ikinci bir filtreden geçirip daha rafine hale getirir.

---

## Parametreler (Pratik Anlamıyla)

| Parametre                       | Ne kontrol eder?                                 |
| ------------------------------- | ------------------------------------------------ |
| `reduction`, `min_mid_channels` | Bottleneck kapasitesi                            |
| `norm` (`gn/bn/none`)           | Bottleneck stabilitesi                           |
| `dilation`                      | Multi-scale bağlam genişliği                     |
| `init_alpha`, `learnable_alpha` | H/W mask gücünün başlangıcı ve öğrenilebilirliği |
| `beta`                          | Ölçeklemeyi ne kadar “attention” yapacağı        |
| `use_spatial_gate`              | Ek uzamsal kapı                                  |

---

## Çıktılar ve İzleme

Bu implementasyon son üretilen maskeleri saklar:

- `attn_h`
- `attn_w`

`last_mask_stats()` ile min/mean/max/std alınarak attention davranışı gözlemlenebilir.

---

## Ne Zaman Kullanılır?

- Detection / segmentation gibi konum hassas görevlerde
- Küçük nesnelerde yönsel bilgi kritikse
- Attention blokları training’de agresifleşip stabilite bozuyorsa
- “Coordinate attention” fikrini daha güçlü denemek istiyorsan

---

## Kısa Özet

CA+ şunları yapar:

1. H ve W eksenlerinde mean+max profilleri çıkarır
2. Local + dilated yön filtreleri ile multi-scale özellik toplar
3. Ortak bottleneck ve refine ile temsil gücünü artırır
4. H/W maskelerini üretip identity ile karıştırır (learnable alpha)
5. Beta ile ölçeklemeyi 1 etrafında yumuşatır
6. İsteğe bağlı spatial gate ile ekstra rafine eder

> “Klasik Coordinate Attention’ın daha güçlü, daha kontrollü ve daha stabil versiyonu.”
