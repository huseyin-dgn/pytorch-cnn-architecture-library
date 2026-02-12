# DropBlock (CNN Regularization)

DropBlock, CNN’lerde klasik Dropout’un zayıf kaldığı **uzamsal (spatial) korelasyon** problemini hedefleyen bir regularization tekniğidir. Klasik dropout tek tek aktivasyonları düşürürken, DropBlock feature map üzerinde **bitişik k×k blokları** maskeleyerek (0’layarak) modeli lokal ipuçlarına aşırı bağımlılıktan uzaklaştırır.

---

## Ne Yapar?

- CNN’in ürettiği **feature map** üzerinde çalışır.
- Eğitim sırasında her forward pass’te rastgele **k×k bölgeleri** drop eder.
- Drop edilen bölgeler 0 olur, kalan bölgeler genelde **keep_ratio** ile ölçeklenir.
- `eval()` modunda devre dışıdır (inference’ta maske uygulanmaz).

---

## Neden Ortaya Çıktı?

CNN feature map’lerinde komşu aktivasyonlar yüksek korelasyonlu olduğu için klasik dropout:

- tekil noktaları düşürse bile,
- yanındaki benzer aktivasyonlar bilgiyi taşımaya devam eder.

DropBlock bu yüzden:

- tek tek noktaları değil,
- **blok halinde** bölgeleri düşürerek
  uzamsal korelasyonu doğrudan kırar ve daha güçlü regularization sağlar.

---

## Hangi Problemi Çözer?

- Overfitting (özellikle sınırlı veri / yüksek kapasite)
- Lokal shortcut öğrenme (modelin küçük patch ipuçlarına aşırı yaslanması)
- CNN’lerde dropout’un düşük etkisi (spatial redundancy)

---

## Nereye Entegre Edilir?

**Önerilen kalıp:** Conv → BatchNorm → Activation → DropBlock

**Residual block içinde en iyi yer:**

- **residual toplama öncesi** (skip ile toplanmadan hemen önce)

**Attention varsa:** Conv → BN → Act → Attention → DropBlock

> DropBlock’u BN’den önce koymak genelde etkisini zayıflatır.  
> DropBlock’u her katmana koymak erken katmanlarda detay kaybına yol açabilir.

---

## Parametreler

- `drop_prob (p)`: hedef drop oranı
  - tipik aralık: **0.05 – 0.30**
- `block_size (k)`: düşürülecek blok boyutu
  - feature map’e göre seçilir (çok küçük feature map’te büyük k agresiftir)
- `schedule (önerilir)`: `drop_prob` değerini eğitim boyunca artırmak
  - başta düşük, sonlara doğru yüksek

---

## Pratik Ayar Önerileri

- Orta/derin stage’lerde kullan (erken stage’lerde dikkat).
- Feature map küçüldükçe `block_size`’ı küçült.
- Overfitting belirginse `drop_prob`’u artır; underfitting olursa azalt.
- En stabil kullanım: **drop_prob schedule** ile.

---

## Kısa Özet

DropBlock:

- eğitim sırasında feature map’te **k×k bölgeleri** rastgele maskeler,
- CNN’lerde uzamsal korelasyonu kırarak dropout’tan daha güçlü regularization sağlar,
- inference’ta kapalıdır ve tam feature map kullanılır.
