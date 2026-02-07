## Octave Convolution

OctaveConv’un olayı: **feature map’i yüksek frekans (H) ve düşük frekans (L) diye iki yola ayırıp** düşük frekansı daha düşük çözünürlükte taşımak.  
Amaç: **hesabı ve bellek kullanımını düşürmek** + düşük frekans bilgiyi daha “geniş bağlam” gibi kullanmak.

### H/L ne demek?

- **H (High frequency):** kenar, detay, ince bilgi → **H×W çözünürlükte**
- **L (Low frequency):** daha yumuşak / global bilgi → **H/2 × W/2 çözünürlükte** (daha ucuz)

Bu ayrımı kanal oranı belirler:

- `alpha` (veya `alpha_in / alpha_out`) = düşük frekansa ayrılan kanal oranı  
  Örn `alpha=0.5` → kanalların yarısı L’ye gider.

### Bu implementasyonda akış nasıl?

Model, her OctaveConv katmanında 4 dönüşüm yapabiliyor:

- **H → H (hh):** yüksekten yükseğe normal conv
- **H → L (hl):** önce H downsample (AvgPool) → sonra conv
- **L → L (ll):** düşük çözünürlükte conv (ucuz)
- **L → H (lh):** L’de conv → sonra upsample (interpolate) → H’ye ekle

Çıkış:

- `y_h` ve `y_l` ayrı ayrı döner.

### Split / Merge ne yapıyor?

Bu model klasik tensor `(B,C,H,W)` ile başlıyor, sonra:

- **split:** kanalı ikiye böler
  - `x_h`: ilk `C_h` kanal (H×W)
  - `x_l`: kalan `C_l` kanal → AvgPool ile (H/2×W/2)
- **merge:** `x_l` upsample edilip `x_h` ile concat edilir → tekrar tek tensor olur

### Bu network’te nerede kullanılıyor?

- `stem` normal conv ile 64 kanal çıkarıyor
- sonra `split` ile H/L ayrılıyor
- `b1` ve `b2` OctaveConvBlock ile H/L birlikte işleniyor
- sonra `merge` + GAP + FC ile sınıflandırma

### Ne işe yarar?

- Daha düşük çözünürlükte L yolunu taşıdığı için **compute daha düşük** olur ✅
- L yolu sayesinde daha global bağlam taşınır ✅
- Bazı görevlerde aynı performansı daha ucuza yakalamayı hedefler 🎯

### Dikkat

- `alpha` çok büyürse (L çok artarsa) detay kaybı yaşanabilir ⚠️
- Çok küçük olursa da faydası azalır.
- Genelde 0.25–0.5 aralığı mantıklı başlangıçtır.
