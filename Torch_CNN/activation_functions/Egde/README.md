# Sigmoid / HardSigmoid & Swish / HardSwish

## Karşılaştırmalı Deneyler – Uygulama ve Sonuç Özeti

Bu çalışma kapsamında, CNN mimarilerinde sık kullanılan iki aktivasyon ailesi **aynı koşullar altında deneysel olarak test edilmiştir**:

- **Sigmoid vs HardSigmoid** (gating / attention kullanımı)
- **Swish (SiLU) vs HardSwish** (backbone activation kullanımı)

Amaç:

- Aktivasyon fonksiyonlarını teorik değil, **pratik performans üzerinden** değerlendirmek
- Aynı mimari ve eğitim koşullarında **gerçek farkları ölçmek**
- Backbone ve attention rollerini **deneysel veriye dayanarak ayırmak**

---

## Deney Kurulumu (Standart)

Tüm karşılaştırmalar şu şekilde yapılmıştır:

- Aynı model mimarisi
- Sadece aktivasyon fonksiyonu değiştirilmiştir
- Aynı optimizer, learning rate ve scheduler
- Aynı dataset ve augmentasyon
- Aynı seed
- Aynı epoch sayısı

Bu sayede gözlemlenen farklar **doğrudan aktivasyon fonksiyonuna** aittir.

---

## 1️⃣ Swish (SiLU) vs HardSwish – Backbone Karşılaştırması

Bu deneyde:

- Swish ve HardSwish **ana aktivasyon** olarak kullanılmış
- Modeller yan yana eğitilmiş
- Epoch bazlı loss ve accuracy loglanmıştır

### Gözlem

- Swish, tüm eğitim boyunca **daha yüksek test accuracy** üretmiştir
- HardSwish daha hızlı yakınsasa da **final performansta geride kalmıştır**
- Fark küçük ama **istikrarlı**dır

### Sonuç

- **Accuracy öncelikli senaryolarda Swish**
- **Mobil / edge / latency öncelikli senaryolarda HardSwish**

Backbone için iki aktivasyon da kullanılabilir; seçim **hedef platforma** bağlıdır.

---

## 2️⃣ Sigmoid vs HardSigmoid – Attention / Gating Karşılaştırması

Bu deneylerde:

- Sigmoid ve HardSigmoid **backbone activation olarak değil**
- SE / attention benzeri **gating mekanizmaları içinde** kullanılmıştır
- Amaç, feature üretmek değil **feature ölçeklemek**tir

### Gözlem

- Her iki aktivasyon da gate çıktısını `[0,1]` aralığında üretmiştir
- HardSigmoid, sigmoid’e çok yakın davranış göstermiştir
- Hesaplama maliyeti HardSigmoid tarafında daha düşüktür

### Sonuç

- Attention / gating için **ikisinin de kullanımı doğrudur**
- Accuracy öncelikli senaryolarda **Sigmoid**
- Mobil / verimlilik öncelikli senaryolarda **HardSigmoid**

Bu aile **ana aktivasyon olarak kullanılmamıştır** ve kullanılmamaktadır.

---

## Net Ayrım (Bu Repoda Benimsenen Yaklaşım)

| Rol                        | Kullanılan Aktivasyon    |
| -------------------------- | ------------------------ |
| Backbone ana aktivasyon    | Swish (SiLU) / HardSwish |
| Attention / SE / Gate      | Sigmoid / HardSigmoid    |
| Output probability         | Sigmoid                  |
| Backbone’da Sigmoid ailesi | ❌ kullanılmaz           |

---

## Genel Sonuç

Bu çalışmalar göstermiştir ki:

- Aktivasyon fonksiyonları **tek başına iyi/kötü değildir**
- Doğru sonuç, **doğru yerde kullanılan aktivasyondan** gelir
- Swish/HardSwish → **temsil gücü**
- Sigmoid/HardSigmoid → **kontrol ve ölçekleme**

Bu repo, bu ayrımı teorik anlatımla değil, **aynı koşullarda yapılan deneylerle** ortaya koymaktadır.

---

📌 Tüm sonuçlar:

- `.ipynb` dosyalarında yeniden üretilebilir
- Loglar epoch bazında kayıt altındadır
- Modeller aynı mimariyi paylaşmaktadır
