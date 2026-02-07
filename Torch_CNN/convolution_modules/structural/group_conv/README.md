## Group Convolution (ResNeXt-style)

GroupConv’un olayı: **kanalları gruplara bölüp konvolüsyonu ayrı ayrı yapmak.**  
Normal conv’da her output kanal, tüm input kanallarıyla bağlıdır.  
Group conv’da bağlantılar “parçalanır” → **hesap azalır**, aynı zamanda **cardinality** (çoklu grup) fikriyle kapasiteyi farklı şekilde artırır. 🧩

### GroupConv ne sağlar?

- **Compute/param düşer:** bağlantı sayısı bölünür
- **Cardinality artar:** “kaç farklı grup/uzman” gibi düşünebilirsin (ResNeXt mantığı)
- Özellikle 3×3 conv maliyetinde ciddi fark eder

---

## Bu kodda ne oluyor?

### 1) CBR

Standart blok:

- `Conv2d(groups=G) + BN + ReLU`
- `groups=1` → normal conv
- `groups=cin` gibi olursa depthwise’a yaklaşır (burada öyle değil)

### 2) GroupResBlock (ResNeXt benzeri bottleneck)

Sıra net:

1. **1×1 reduce**  
   `cin -> mid` (kanalı düşürüp mixing yapar)

2. **3×3 group conv (asıl olay)**  
   `mid -> mid` ama `groups=groups`  
   stride burada olabiliyor (downsample)

3. **1×1 expand**  
   `mid -> cout` (tekrar büyütür)

4. **Skip (residual)**
   - `stride=1` ve `cin=cout` ise identity
   - değilse 1×1 ile eşleştirip toplar

Bu, klasik “bottleneck + residual” şablonunun group conv’lu hali.

> Not: Kodda `mid % groups != 0` ise groups’u düşürüp uygun hale getiriyor. Çünkü group conv’da `in_channels % groups == 0` şart.

### 3) GroupConvNet

- Stem: başta 2 tane normal conv (grupsuz) → erken mixing güçlü kalsın diye
- Stage1/2/3:
  - ilk blok stride=2 ile downsample
  - kalan bloklar stride=1
- Son: GAP + FC

---

## Ne işe yarar?

- ResNeXt mantığı: “daha geniş değil, **daha çok grup**” ile representasyonu güçlendirmek 🎯
- Parametreyi şişirmeden kapasiteyi farklı yönden artırır
- Büyük görüntü boyutlarında compute avantajı net çıkar

---

## Dikkat

- `groups` büyüdükçe compute düşer ama **kanallar arası etkileşim azalır**.
  Bu yüzden:
  - 1×1 conv’lar (reduce/expand) kritik: mixing’i onlar geri getiriyor ✅
