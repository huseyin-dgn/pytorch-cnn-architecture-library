# Attention–Residual Fusion Pattern (CBAM Tabanlı)

Bu mimaride kullanılan pattern, **residual öğrenme** ile **attention odaklanmasını** doğrudan birleştirir. Yani ağ hem derinliği stabil şekilde öğrenir hem de önemli özellikleri seçici biçimde güçlendirir.

---

## 🎯 Temel Formül

**Çıkış = Aktivasyon( Skip(x) + A(F(x)) ⊙ F(x) )**

Burada:

- **F(x)** → Konvolüsyonlardan geçen residual ana yol
- **A(·)** → Attention haritası
- **⊙** → Eleman bazlı çarpım (feature yeniden ağırlıklandırma)
- **Skip(x)** → Kimlik veya projeksiyon kısa yolu

Attention, residual özellikleri **toplama öncesi filtreler**.

---

## 🧠 Kullanılan Attention Türü: CBAM

Bu yapı **CBAM (Convolutional Block Attention Module)** kullanır ve iki aşamalıdır:

### 1️⃣ Channel Attention

Model şuna karar verir:

> “Hangi feature kanalları önemli?”

Global ortalama ve maksimum havuzlama ile kanal önem haritası çıkarılır ve feature’lar kanal bazında ölçeklenir.

### 2️⃣ Spatial Attention

Model şuna karar verir:

> “Hangi uzamsal bölgeler önemli?”

Kanallar üzerinden özetlenmiş haritalarla uzamsal dikkat maskesi üretilir ve feature haritası piksel düzeyinde filtrelenir.

---

## 🔀 Fusion Noktası

Attention, residual blokta şu noktada uygulanır:

**Conv → Norm → Conv → Norm → CBAM → Toplama**

Yani önce residual özellik üretilir (**F(x)**), sonra attention ile yeniden ağırlıklandırılır.

---

## 🔁 Residual Yapı Korunur

Attention eklenmesine rağmen:

✔ Skip yolu aynen durur  
✔ Toplama işlemi değişmez  
✔ Gradyan akışı residual hat üzerinden stabil kalır

Attention sadece ana yolun temsilini “daha seçici” yapar.

---

## 🧩 Basic ve Bottleneck Farkı

| Blok Türü      | Ana Yol Yapısı  |
| -------------- | --------------- |
| **Basic**      | 3×3 → 3×3       |
| **Bottleneck** | 1×1 → 3×3 → 1×1 |

Her iki durumda da **attention, son konv sonrası ve toplama öncesi** yer alır.

---

## 🚀 Bu Pattern Ne Sağlar?

✔ Gürültülü özellikleri bastırır  
✔ Önemli kanalları ve bölgeleri güçlendirir  
✔ Residual stabilite korunur  
✔ Sınıflandırma, detection ve segmentation’da performans artışı sağlar

---

## 🔚 Özet

Bu pattern:

**Residual öğrenme + Kanal dikkati + Uzamsal dikkat**

kombinasyonudur ve modern CNN’lerde en etkili feature iyileştirme yöntemlerinden biridir.
