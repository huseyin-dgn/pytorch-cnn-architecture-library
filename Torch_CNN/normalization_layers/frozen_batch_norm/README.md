# Frozen BatchNorm (FrozenBatchNorm2d)

## Nedir?

**FrozenBatchNorm2d**, klasik `BatchNorm2d` gibi normalize eder ama kritik bir farkla:

- **Batch’ten yeni istatistik (mean/var) hesaplamaz**
- **running_mean / running_var güncellenmez**
- Eğitim modunda bile davranış **eval BN** gibidir (popülasyon istatistikleriyle)

Yani:

> **BN:** training’de batch istatistiği, eval’de running istatistiği  
> **FrozenBN:** her zaman running istatistiği

Bu yüzden “frozen” denir: istatistikler “kilitlenmiştir”.

---

## Neden kullanılır?

### 1) Küçük batch sorununu azaltmak

BatchNorm’un en büyük problemi küçük batch’lerde istatistiğin gürültülü olmasıdır.  
FrozenBN bu gürültüyü ortadan kaldırır çünkü batch’e bakmaz.

### 2) Fine-tuning / transfer learning senaryosu

Pretrained modeldeki BN istatistikleri “iyi” olabilir.  
Yeni dataset küçükse veya domain kayıyorsa BN güncellenmesi bazen performansı bozar.

FrozenBN yaklaşımı:

- BN’i “kapat”
- modelin geri kalanını fine-tune et

### 3) Deterministik davranış

BN training modunda batch’e göre değişir.  
FrozenBN ile aynı input → daha stabil output.

---

## Bu implementasyonda FrozenBN ne yapıyor?

`FrozenBatchNorm2d` şu buffer’ları tutuyor:

- `running_mean`
- `running_var`
- (affine ise) `weight` (gamma), `bias` (beta)

Forward’da:

1. `x` running istatistiklerle normalize edilir
2. affine açıksa gamma/beta uygulanır

> Batch’ten mean/var çıkarılmaz, hiçbir şey güncellenmez.

---

## “BN → FrozenBN” dönüşümü mantığı

Bu repoda yaklaşım şöyle:

1. Model normal `BatchNorm2d` ile kurulur
2. (İstersen) BN’ler eğitimde çalıştırılır → running_mean/var oluşur
3. Sonra tüm `BatchNorm2d` katmanları **FrozenBatchNorm2d** ile değiştirilir
4. Bu andan itibaren model BN güncellemez

Bu dönüşümü yapan yardımcı fonksiyon:

- `convert_bn_to_frozen(model)`

Bu fonksiyon:

- modelin tüm alt modüllerini dolaşır
- `BatchNorm2d` görürse FrozenBN üretir
- running stats ve varsa gamma/beta’yı **kopyalar**
- sonra o katmanı replace eder

**Önemli nokta:**  
FrozenBN “bir şey öğrenmez”; sadece **hazır istatistikleri kullanır**.  
Bu yüzden doğru kullanımda BN’in running stats’i “anlamlı” olmalıdır.

---

## Bu klasördeki örnek model neyi gösteriyor?

Verilen örnek ağ:

- Conv → Norm → Act blokları
- Bottleneck + SE
- stage’ler (ResNet benzeri)
- en sonda head norm + classifier

Amaç:

- FrozenBN’in sadece “tek layer” değil, gerçek bir backbone içinde nasıl kullanılacağını göstermek.

---

## Ne zaman tercih edilir?

✅ Küçük batch (özellikle GPU kısıtlı)  
✅ Fine-tuning (pretrained backbone)  
✅ Domain shift / istatistik güncellemesi kararsızsa  
✅ Multi-GPU sync yoksa ama BN stabil olsun istiyorsan

❌ Baştan sıfırdan training (özellikle büyük batch)  
❌ running stats hiç oluşmamışsa / yanlışsa  
❌ Dataset çok farklı ama BN’i sabitlemek zarar veriyorsa

---

## Pratik öneri

FrozenBN genelde şu şekilde kullanılır:

- **Aşama-1:** BN ile kısa bir süre train (ya da pretrained istatistik kullan)
- **Aşama-2:** BN → FrozenBN çevir
- **Aşama-3:** Fine-tune (özellikle head / üst katmanlar)

---

## Kısa Özet

FrozenBatchNorm2d:

- BatchNorm gibi normalize eder
- Ama batch’ten istatistik almaz ve running stats güncellemez
- Küçük batch ve fine-tuning senaryolarında stabil bir alternatiftir
- BN katmanlarını “donmuş istatistiklerle” deterministik hale getirir
