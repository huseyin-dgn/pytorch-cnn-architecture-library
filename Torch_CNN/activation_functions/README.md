# Activation Functions – Karşılaştırmalı Deneyler ve Seçim Rehberi

Bu dizin, CNN mimarilerinde kullanılan **aktivasyon fonksiyonlarının** teorik arka planını, pratik kullanım alanlarını ve **aynı koşullarda yapılan deneysel karşılaştırmalarını** içerir.

Amaç:

- Aktivasyon fonksiyonlarının **gerçek model performansına etkisini** göstermek
- “Hangisi daha iyi?” sorusuna **sayısal kanıtla** cevap vermek
- Backbone, attention ve deploy senaryoları için **net seçim kuralları** sunmak

---

## Kapsanan Aktivasyonlar

- ReLU
- LeakyReLU
- PReLU
- SiLU (Swish)
- HardSwish
- Sigmoid / HardSigmoid (özellikle attention & gating için)

---

## Deney Ortamı (Standartlaştırılmış)

Tüm karşılaştırmalar:

- **Aynı mimari**
- **Aynı optimizer, LR, scheduler**
- **Aynı dataset ve augmentasyon**
- **Aynı seed**
- **Aynı epoch sayısı**

kullanılarak yapılmıştır.

Bu sayede farklar **yalnızca aktivasyon fonksiyonundan** kaynaklanmaktadır.

---

## 📊 Aktivasyon Karşılaştırması – Kesin Sonuç

### Final Epoch (15/15)

| Activation | Final Train Loss | Final Test Loss | Final Test Accuracy |
| ---------- | ---------------- | --------------- | ------------------- |
| ReLU       | 0.4770           | 0.6740          | 0.7797              |
| SiLU       | 0.4712           | 0.5892          | 0.8029              |
| LeakyReLU  | 0.5041           | 0.5645          | 0.8088              |
| **PReLU**  | **0.4614**       | **0.5573**      | **0.8149**          |

### Peak Accuracy (Tüm Eğitim Boyunca)

| Activation | Peak Accuracy | Epoch  |
| ---------- | ------------- | ------ |
| ReLU       | 0.7915        | 14     |
| SiLU       | 0.8102        | 14     |
| LeakyReLU  | 0.8088        | 15     |
| **PReLU**  | **0.8149**    | **15** |

---

## 🏆 Genel Değerlendirme

### 1️⃣ PReLU – En Güçlü Performans

- En yüksek **final accuracy**
- En düşük **test loss**
- Eğitim ilerledikçe performans düşmüyor
- Kanal başına öğrenilebilir negatif eğim sayesinde **daha esnek temsil**

➡️ **Accuracy öncelikli projeler için birincil tercih**

---

### 2️⃣ LeakyReLU – Dengeli ve Hafif

- Parametresiz
- Stabil eğitim
- PReLU’ya çok yakın performans
- Deploy / sade mimariler için avantajlı

➡️ **Lightweight ve production senaryoları için ideal**

---

### 3️⃣ SiLU (Swish) – Modern ama bu deneyde üçüncü

- Düzgün gradyan akışı
- ReLU’dan açıkça iyi
- Ancak bu deneyde **PReLU ve LeakyReLU’nun gerisinde**

➡️ **Alternatif olarak kullanılabilir ama varsayılan tercih değil**

---

### 4️⃣ ReLU – En Zayıf Seçenek

- Dying ReLU problemi
- Test loss dalgalı
- Final accuracy en düşük

➡️ **Bu mimari ve ayarlarda önerilmez**

---

## HardSigmoid & HardSwish Notu (Özel Kullanım)

Bu aktivasyonlar backbone için değil, **gating ve efficiency** odaklıdır:

- **HardSigmoid**
  - SE / CBAM / Coordinate Attention gibi **attention gate**’lerde uygundur
  - 0–1 aralığında ölçekleme yapar
- **HardSwish**
  - Mobil / edge modellerde
  - Swish’e yakın performans + daha düşük hesaplama maliyeti

---

## 🔧 Pratik Seçim Rehberi

| Senaryo                     | Önerilen Aktivasyon   |
| --------------------------- | --------------------- |
| Accuracy öncelikli CNN      | **PReLU**             |
| Dengeli / sade mimari       | LeakyReLU             |
| Modern backbone (opsiyonel) | SiLU                  |
| Mobil / edge backbone       | HardSwish             |
| Attention / gating          | Sigmoid / HardSigmoid |
| Klasik ReLU                 | ❌ önerilmez          |

---

## Sonuç (Tek Cümle)

> Bu deneylerde **PReLU**, CNN backbone’ları için en yüksek doğruluk ve en iyi genelleme performansını sağlamış; **LeakyReLU** ise deploy açısından en dengeli alternatif olmuştur.

---

📌 Bu dizindeki `.ipynb` ve `models/` klasörleri, yukarıdaki sonuçların **doğrudan yeniden üretilebilir** kodlarını içerir.
