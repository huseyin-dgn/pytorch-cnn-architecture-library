# CNN Regularization Teknikleri — Genel Karşılaştırma

Bu doküman aşağıdaki tekniklerin karşılaştırmasını içerir:

- DropBlock
- SpatialDropout
- DropPath
- Stochastic Depth
- Cutout

Amaç: Hangi teknik nerede kullanılır, neyi çözer ve nasıl farklılaşır sorularına net cevap vermek.

---

# 1) Temel Çalışma Alanı

| Teknik           | Uygulama Seviyesi                  |
| ---------------- | ---------------------------------- |
| Cutout           | Input görüntü                      |
| SpatialDropout   | Feature map (kanal bazlı)          |
| DropBlock        | Feature map (uzamsal blok)         |
| Stochastic Depth | Residual branch                    |
| DropPath         | Residual branch (genelleştirilmiş) |

---

# 2) Ne Tür Bağımlılığı Kırar?

| Teknik           | Kırdığı Bağımlılık        |
| ---------------- | ------------------------- |
| Cutout           | Lokal görüntü bölgesi     |
| SpatialDropout   | Belirli feature kanalı    |
| DropBlock        | Uzamsal aktivasyon kümesi |
| Stochastic Depth | Belirli residual blok     |
| DropPath         | Residual yol              |

---

# 3) Agresiflik Seviyesi

| Teknik           | Agresiflik                   |
| ---------------- | ---------------------------- |
| SpatialDropout   | Düşük-Orta                   |
| Cutout           | Orta                         |
| DropBlock        | Orta-Yüksek                  |
| Stochastic Depth | Orta                         |
| DropPath         | Orta-Yüksek (schedule varsa) |

---

# 4) Classification İçin Uygunluk

| Teknik           | Classification |
| ---------------- | -------------- |
| Cutout           | Çok uygun      |
| SpatialDropout   | Uygun          |
| DropBlock        | Uygun          |
| Stochastic Depth | Uygun          |
| DropPath         | Uygun          |

---

# 5) Detection İçin Uygunluk

| Teknik           | Detection |
| ---------------- | --------- |
| Cutout           | Riskli    |
| SpatialDropout   | Orta      |
| DropBlock        | Uygun     |
| Stochastic Depth | Uygun     |
| DropPath         | Çok uygun |

---

# 6) Multi-scale Sistemlerde (FPN + Detection)

| Teknik           | Multi-scale Uyum |
| ---------------- | ---------------- |
| Cutout           | Düşük            |
| SpatialDropout   | Orta             |
| DropBlock        | Yüksek           |
| DropPath         | Yüksek           |
| Stochastic Depth | Orta-Yüksek      |

---

# 7) Mühendislik Perspektifi

## Input Seviyesi Regularization

- Cutout
- CutMix (bu tabloda yok ama benzer sınıf)

Amaç:
Modeli eksik veriyle öğrenmeye zorlamak.

---

## Feature Seviyesi Regularization

- SpatialDropout
- DropBlock

Amaç:
Feature bağımlılığını kırmak.

---

## Mimari Seviyesi Regularization

- Stochastic Depth
- DropPath

Amaç:
Derinlik bazlı regularization sağlamak.

---

# 8) Hangi Durumda Hangisi?

### Küçük dataset + classification

→ Cutout veya SpatialDropout

### Orta derin CNN + overfitting

→ DropBlock

### Çok derin residual mimari

→ DropPath veya Stochastic Depth

### Modern backbone (EfficientNet, ViT benzeri)

→ DropPath

### Detection + FPN

→ DropBlock + DropPath kombinasyonu mantıklı

---

# 9) Sezgisel Özet

Cutout:
Giriş seviyesinde müdahale eder.

SpatialDropout:
Kanal bağımlılığını kırar.

DropBlock:
Uzamsal yapıyı bilinçli bozar.

Stochastic Depth:
Blok seviyesinde rastgele derinlik azaltır.

DropPath:
Modern ve esnek residual regularization yöntemidir.

---

# Sonuç

Bu teknikler aynı amaca hizmet eder:
Overfitting azaltmak ve genelleme artırmak.

Ancak:

- Uygulama seviyeleri farklıdır.
- Agresiflikleri farklıdır.
- Detection ve classification uyumlulukları farklıdır.

Seçim, model mimarisi ve problem türüne göre yapılmalıdır.
