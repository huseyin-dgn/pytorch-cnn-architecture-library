# CBAM Channel + Coordinate Attention Plus (Temperature + Safety-Gated Residual)

## Amaç

Bu blok, **CBAM tabanlı gelişmiş Channel Attention** ile **CoordinateAttPlus** yapısını birleştirir ve attention uygulamasını **güvenli residual karışım** ile stabilize eder.

Hedef 3 şey:

1. Kanal önemini **dinamik ve kontrollü** üretmek
2. Konum/yön bilgisini kaybetmeden **coordinate tabanlı** ölçekleme yapmak
3. Training sırasında attention’ın aşırı bastırma yapmasını engelleyip **kendini toparlayan (kurtarma)** bir residual mekanizması eklemek

---

## Bu Yapı Neyi Farklı Yapıyor?

| Bileşen              | Standart yaklaşımlar   | Bu blok                                             |
| -------------------- | ---------------------- | --------------------------------------------------- |
| Channel Attention    | Avg+Max sabit birleşim | **Router + sıcaklık (Tr) ile yumuşak/sert seçim**   |
| Channel gating       | Sigmoid sabit          | **T ile keskinlik kontrolü + clamp (t_min, t_max)** |
| CA ölçekleme         | Direkt `x * ca`        | **Beta kontrollü ölçek: 1 + beta_ca\*(ca-1)**       |
| Coordinate attention | Basit eksen pooling    | **Yerel + dilated yön filtreleri + clamp**          |
| Residual             | Sabit alpha            | **Alpha temiz + alpha etkin (monitor/kurtarma)**    |

---

## 1) ChannelAttentionFusionT (Gelişmiş Kanal Dikkati)

### Fusion (Avg/Max birleştirme)

Bu modül, Avg ve Max istatistiklerini her örnek için dinamik bir şekilde birleştirir:

- `fusion="sum"`: klasik toplama
- `fusion="softmax"`: router ile **öğrenilen ağırlıklar**

**Router temperature (Tr):**  
Softmax’ın ne kadar “keskin” karar vereceğini kontrol eder.

- Tr ↓ → daha agresif seçim
- Tr ↑ → daha yumuşak karışım

### Temperature gating + clamp

Kanal attention üretiminde sıcaklık parametresi kullanılır ve **t_min–t_max** aralığında clamp edilir.

Bu iki şeyi sağlar:

- Çok keskin attention yüzünden patlamayı azaltır
- Çok yumuşak attention yüzünden etkisizleşmeyi azaltır

### Beta kontrollü kanal ölçekleme

Klasik yaklaşım `x * ca` iken burada:

> ölçekleme 1 etrafında “yumuşatılır”

Bu, attention’ın aşırı baskılamasını azaltıp daha stabil bir etki üretir.

---

## 2) CoordinateAttPlus (Clamp + Multi-Scale Directional CA)

CoordinateAttPlus, yönsel (H/W) profiller üzerinden attention üretir ve iki kritik geliştirme içerir:

### Multi-scale yön filtreleme

- Yerel depthwise yön filtreleri
- Dilated depthwise yön filtreleri

Bu sayede:

- Kısa menzilli yapılar
- Daha geniş bağlam

aynı anda yakalanabilir.

### Ölçek clamp (scale_min–scale_max)

Coordinate ölçekleme, belirli bir aralıkta tutulur:

- Aşırı büyütme → engellenir
- Aşırı bastırma → engellenir

Bu, özellikle training stabilitesi için önemli.

### Öğrenilebilir alpha (eksensel karışım)

H ve W attention’larının etkisi öğrenilebilir şekilde ayarlanır.
Yani model şunu öğrenir:

> “Bu veri için yatay mı daha önemli, dikey mi?”

---

## 3) Residual + “Kurtarma” Mekanizması (Monitor Mode)

Bu blokta residual karışım klasik değil:

- `alpha_temiz`: temel karışım katsayısı (0..1)
- `alpha_etkin`: training sırasında otomatik ayarlanan katsayı

### Neden?

Bazen attention blokları çıktıyı aşırı bastırır ve “enerji” düşer.  
Bu durumda model öğrenmeyi kaybedebilir.

### Çözüm: enerji oranına göre alpha kısma

Model, çıktı enerji oranını (std/ratio) izler ve `r_ema` ile EMA tutar.  
Eğer enerji düşüyorsa:

- `alpha_etkin` düşürülür (attention etkisi azaltılır)
- Çıkış tekrar “kurtarılır”

### Kurtarma Modları

- `ratio_floor`: belirli bir minimum oranı garanti eder
- `alpha_floor`: alpha için minimum etki sınırı koyar

Bu mekanizma “attention yüzünden modelin boğulmasını” engeller.

---

## Bu Yapı Ne Tür Attention?

| Tür                       | Var mı?                          |
| ------------------------- | -------------------------------- |
| Channel Attention         | ✔ (dinamik fusion + temperature) |
| Coordinate Attention      | ✔ (multi-scale directional)      |
| Hybrid Attention          | ✔                                |
| Residual Gated Attention  | ✔                                |
| Safety/Recovery Mechanism | ✔ (monitor + EMA)                |

---

## Ne Zaman Kullanılır?

- Detection / segmentation gibi zor görevlerde
- Küçük nesne + karmaşık arka plan senaryolarında
- Attention blokları agresifleşip stabilite bozuyorsa
- Mimari araştırmalarında “kontrollü attention” deneniyorsa

---

## Kısa Özet

Bu blok:

1. Avg/Max kanal istatistiklerini router ile dinamik birleştirir
2. Temperature + clamp ile attention keskinliğini kontrol eder
3. Beta ile kanal ölçeklemeyi yumuşatır
4. CoordinateAttPlus ile yönsel + multi-scale konum bilgisi ekler
5. Monitor/kurtarma ile residual karışımı training sırasında otomatik stabilize eder

> Bu, CBAM kanal dikkatini + coordinate dikkatini **stabil ve güvenli** biçimde birleştiren araştırma seviyesinde bir bloktur.
