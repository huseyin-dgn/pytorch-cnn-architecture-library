# 🔺 Feature Pyramid Network (FPN)

## 📌 Amaç

Feature Pyramid Network (FPN), farklı çözünürlüklerdeki feature map'leri birleştirerek **multi-scale temsil** üretmek için kullanılan bir neck yapısıdır.

Özellikle:

- Object Detection
- Instance Segmentation
- Küçük + orta + büyük objelerin birlikte bulunduğu problemler

için kritik öneme sahiptir.

---

## 🧠 Neden FPN?

Backbone içindeki derin katmanlar:

- Daha güçlü semantik bilgi taşır
- Ancak düşük çözünürlüktedir

Erken katmanlar:

- Yüksek çözünürlüktedir
- Ancak semantik olarak zayıftır

FPN bu iki bilgiyi birleştirir.

---

## 🏗️ Mimari Yapı

FPN üç ana adımdan oluşur:

### 1️⃣ Lateral Connection (1x1 Conv)

Backbone’dan gelen C2–C5 feature map’lerinin kanal sayısı eşitlenir.

- C2, C3, C4, C5 → L2, L3, L4, L5

Amaç: Kanal hizalama (channel alignment)

---

### 2️⃣ Top-Down Pathway (Upsample + Add)

En üst seviyeden başlanır:

```text
P5 = L5
P4 = L4 + Upsample(P5)
P3 = L3 + Upsample(P4)
P2 = L2 + Upsample(P3)
```

Amaç:

- Üst seviyedeki semantik bilgiyi alta taşımak
- Yüksek çözünürlük + güçlü semantik üretmek

---

### 3️⃣ Smoothing (3x3 Conv)

Her P seviyesi 3x3 conv ile temizlenir: **P2, P3, P4, P5**

Amaç:

- Upsample sonrası oluşabilecek artefact’ları azaltmak
- Feature kalitesini stabilize etmek

---

## 📊 Seviye Anlamları

| Seviye | Yaklaşık Stride | Kullanım      |
| ------ | --------------- | ------------- |
| P2     | 4               | Küçük objeler |
| P3     | 8               | Küçük-Orta    |
| P4     | 16              | Orta-Büyük    |
| P5     | 32              | Büyük         |

Head detection gibi küçük obje yoğun problemlerde **P2 kritik** seviyedir.

---

## ⚙️ Ne Zaman Kullanılmalı?

### ✅ Kullan

- Multi-scale object detection
- Küçük objeler mevcutsa
- Aynı görüntüde farklı boyutlarda objeler varsa

### ❌ Gerekli Değil

- Sadece classification
- Tek ölçekli obje problemi
- Edge cihazlarda çok hafif model gerekiyorsa

---

## ⚖️ Avantajlar

- Multi-scale performans artışı
- Küçük obje başarısı
- Daha stabil detection

## ⚠️ Dezavantajlar

- Ek compute maliyeti
- Daha fazla parametre
- Yanlış kullanıldığında overfitting riski

---

## 🧪 Deney Önerisi

FPN kullanırken şu kıyası yap:

- C3–C5 (3 seviye)
- C2–C5 (4 seviye)

Eğer küçük objeler önemliyse C2 eklemek genelde mAP artırır.

---

## 🎯 Özet

FPN bir “zorunluluk” değil, bir **araçtır**.

Problem multi-scale ise:

> FPN mantıklıdır.

Problem tek ölçekliyse:

> Gereksiz karmaşıklık olabilir.

Doğru karar veri dağılımına göre verilmelidir.
