# Batch Renormalization (BRN)

## Nedir?

**Batch Renormalization (BatchRenorm / BRN)**, Batch Normalization’ın (BN) eğitim sırasında **mini-batch istatistiklerine aşırı bağımlı** olmasından doğan problemi hedefler.

BN eğitimde her batch’in ortalama/variance’ını kullanır; batch küçükse veya dağılım dalgalanıyorsa model “zıplar”.  
BRN, BN’in eğitim davranışını **running mean/var (popülasyon istatistikleri)** ile daha uyumlu hale getirir.

Kısaca:

> **BN:** “batch ne diyorsa o”  
> **BRN:** “batch’i kullan ama running istatistiklerden çok sapmasına izin verme”

---

## Mantık (BN → BRN geçişi)

### 1) BN’in standart normalize etmesi

Eğitimde:

- `mean_batch`, `std_batch` hesaplanır
- `x_hat = (x - mean_batch) / std_batch`

Bu tamamen batch’e bağlıdır.

### 2) BRN’in düzeltme (renorm) adımı

BRN iki düzeltme katsayısı kullanır:

- **r**: ölçek düzeltmesi
- **d**: kaydırma düzeltmesi

Bu katsayılar batch istatistikleri ile running istatistiklerin farkından türetilir ve **gradient dışı** (detach) tutulur.

Amaç:

- Eğitimdeki normalize edilmiş aktivasyonları,
- Eval’de kullanılan running istatistiklere yaklaştırmak.

---

## r ve d nedir?

- `r = std_batch / std_running`
- `d = (mean_batch - mean_running) / std_running`

Bu ikisi normalize edilmiş çıktıya uygulanır:

> `x_hat = x_hat * r + d`

### Neden `detach`?

Çünkü bu katsayılar bir “öğrenilecek parametre” değil, **istikrarlı bir düzeltme kuralı**.  
Modelin gradient ile “r ve d’ye göre hile yapması” istenmez.

---

## Clamp (rmax, dmax) ve Warmup

BRN’in kritik tarafı şurası:

- r ve d başta **çok agresif** olursa eğitim bozulabilir.

Bu yüzden:

- `r` değeri `[1/rmax, rmax]` içinde tutulur
- `d` değeri `[-dmax, dmax]` içinde tutulur

Ayrıca bu sınırlar bir anda açılmaz:

### Warmup

`warmup_steps` süresince rmax ve dmax kademeli artar:

- başta: `rmax≈1`, `dmax≈0` → davranış BN’e yakın
- zamanla: hedef değerlere yaklaşır → BRN etkisi artar

Bu, özellikle eğitim başlangıcında stabilite sağlar.

---

## Eğitim / Değerlendirme Davranışı

### Train modunda

- Batch mean/var hesaplanır
- r ve d (running ile) çıkarılır + clamp uygulanır
- running mean/var momentum ile güncellenir

### Eval modunda

- BN gibi doğrudan **running mean/var** ile normalize edilir
- `track_running_stats=False` ise eval’de çalışmaz (bu implementasyonda bilinçli olarak hata veriyor)

---

## Bu klasörde ne var?

### `BatchRenorm2d`

- BN drop-in alternatifi gibi kullanılır
- `rmax`, `dmax`, `warmup_steps` ile kontrol edilir
- `affine=True` ise (gamma/beta) parametreleri vardır

### Örnek kullanım blokları

- **ConvBRNAct:** Conv → BRN → SiLU
- **ResidualBRNBlock:** BRN’li residual blok
- **BRNBackbone:** Çok aşamalı BRN backbone
- **BRNClassifier:** Basit sınıflandırıcı örneği (backbone + GAP + FC)

Amaç: BRN’in sadece “tek layer” değil, gerçek bir backbone içinde nasıl konumlandırılacağını göstermek.

---

## Ne Zaman BRN Mantıklı?

✅ Batch size küçükken (özellikle 1–8 arası)  
✅ Dağılımı dalgalı / augment yoğun eğitimlerde  
✅ BN yüzünden training instabil oluyorsa  
✅ Train/Eval davranış farkı yüzünden performans düşüyorsa

❌ Batch büyük ve stabilse (BN zaten iyi çalışır)  
❌ LayerNorm / GroupNorm gibi batch bağımsız normlar tercih ediliyorsa

---

## Parametre Önerileri (Pratik)

- `rmax`: 2.0–3.0 aralığı iyi başlangıç
- `dmax`: 3.0–5.0 aralığı sık kullanılır
- `warmup_steps`: dataset/batch sayısına göre (ör. 2k–10k)

> Küçük batch’te BRN genelde BN’den daha stabil sonuç verir; ancak her problemde “kesin daha iyi” değildir.

---

## Kısa Özet

Batch Renorm:

- BN’i batch’e fazla bağımlılıktan kurtarır
- running istatistiklerle uyumlu bir normalize davranışı sağlar
- r/d clamp + warmup ile stabiliteyi korur
- küçük batch eğitimlerinde güçlü bir alternatiftir
