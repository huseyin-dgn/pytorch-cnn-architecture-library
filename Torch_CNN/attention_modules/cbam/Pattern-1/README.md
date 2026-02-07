# CBAM Residual Dynamic Spatial Attention

## Amaç

Bu yapı, klasik **CBAM** mekanizmasını geliştirerek daha esnek ve dinamik bir dikkat üretimi sağlar.

Standart CBAM:

- Channel Attention
- Spatial Attention

Bu versiyon ise şunları ekler:

✔ Kanal dikkatinde **öğrenilebilir fusion**
✔ Uzamsal dikkat için **dinamik kernel seçimi**
✔ Dikkat haritalarında **sıcaklık kontrollü ölçekleme**
✔ Residual kontrollü attention uygulaması

Bu nedenle bu blok, klasik CBAM'in genişletilmiş bir araştırma varyasyonudur.

---

## Mimari Bileşenler

### 1) ChannelAttentionFusionT

Klasik CBAM’de:

- Global Average Pooling
- Global Max Pooling
- Sonuçlar toplanır

Bu modülde ise:

| Özellik              | Açıklama                                    |
| -------------------- | ------------------------------------------- |
| Fusion yöntemi       | Toplama veya **Softmax ağırlıklı birleşim** |
| Öğrenilebilir fusion | Avg & Max katkısı öğrenilir                 |
| Sıcaklık (T)         | Attention keskinliği ayarlanabilir          |
| Gate                 | Sigmoid veya HardSigmoid                    |

Bu yapı, kanal dikkatinin ne kadar keskin veya yumuşak olacağını kontrol etmeyi sağlar.

---

### 2) DynamicSpatialAttention

Standart CBAM’de tek bir 7×7 konvolüsyon bulunur.  
Bu yapı ise:

✔ Çoklu kernel boyutları  
✔ Opsiyonel dilated branch  
✔ Router ile dinamik kernel seçimi  
✔ Her giriş için farklı uzamsal dikkat

Bu sayede model şunu öğrenir:

> “Bu görüntü için hangi uzamsal filtre daha uygun?”

---

### 3) Residual Attention Entegrasyonu

Dikkat doğrudan uygulanmaz. Bunun yerine:
out = x + α (y − x)

Bu yapı:

| α değeri      | Etki                   |
| ------------- | ---------------------- |
| α = 1         | Tam attention          |
| α < 1         | Yumuşatılmış attention |
| Öğrenilebilir | Model karar verir      |

---

## Bu Yapı Ne Tür Attention’dır?

| Tür                      | Durum |
| ------------------------ | ----- |
| Channel Attention        | ✔     |
| Spatial Attention        | ✔     |
| Multi-scale Spatial      | ✔     |
| Dynamic Attention        | ✔     |
| Hybrid Attention         | ✔     |
| Residual Gated Attention | ✔     |

---

## Klasik CBAM’den Farkı

| Standart CBAM         | Bu Versiyon           |
| --------------------- | --------------------- |
| Avg+Max sabit toplama | Softmax fusion        |
| Tek kernel spatial    | Çoklu kernel + router |
| Sabit sigmoid         | Sıcaklık kontrollü    |
| Doğrudan çarpım       | Residual kontrollü    |

---

## Avantajları

✔ Daha adaptif dikkat üretir  
✔ Multi-scale uzamsal analiz yapar  
✔ Kanal etkileşimi öğrenilebilir  
✔ Attention şiddeti kontrol edilebilir

---

## Ne Zaman Kullanılır?

- Detection modelleri
- Zor sahneler (karmaşık arka plan)
- Küçük nesne tespiti
- Dikkat varyasyonlarının araştırıldığı çalışmalar

---

## Not

Bu modül klasik bir "ek blok" değil, **attention tasarımı araştırması**dır.  
Farklı fusion, sıcaklık ve kernel kombinasyonları ile varyasyonlar üretilebilir.
