# Basic Residual Blok Tabanlı MiniResNet — Kullanılan Residual Yaklaşım

Bu mimaride klasik **Basic Residual (ResNet-18/34 tipi)** blok yapısı kullanılır. Amaç, derinlik arttıkça oluşan **degradation problemi** ve **gradyan zayıflamasını** engelleyerek stabil öğrenme sağlamaktır.

---

## 🎯 Temel Prensip

Her blok şu işlemi yapar:

**Çıkış = Aktivasyon( F(x) + Skip(x) )**

Burada:

- **F(x)** → Konvolüsyonlardan geçen öğrenen ana yol
- **Skip(x)** → Girişi doğrudan taşıyan kısa yol

Toplama sayesinde ağ, katman ekledikçe performans kaybetmez.

---

## 🧩 Basic Residual Yapısı

Bu blok, bottleneck değil, **klasik iki 3×3 konvolüsyonlu** residual tasarımıdır.

Ana yol:

1. **3×3 Conv → Norm → ReLU**
2. **3×3 Conv → Norm**
3. **(Opsiyonel Attention)**
4. **Skip ile toplama**
5. **ReLU**

Bu yapı uzamsal bilgiyi güçlü şekilde işlerken residual bağlantı bilgi kaybını önler.

---

## 🔁 Skip (Kısa Yol) Davranışı

| Durum                        | Ne Olur                       |
| ---------------------------- | ----------------------------- |
| Kanal ve çözünürlük aynı     | Giriş direkt eklenir          |
| Kanal farklı veya stride ≠ 1 | 1×1 projeksiyon ile eşitlenir |

Bu sayede toplama öncesi tensör boyutları uyumlu hale getirilir.

---

## ⬇️ Downsample Mekanizması

Stage geçişlerinde çözünürlük düşürmek için:

- Ana yolda stride kullanılır
- Skip yol da aynı stride ile projeksiyon yapar

Bu sayede hem uzamsal boyut hem kanal sayısı yeni stage’e taşınır.

---

## 🧠 Normalization Esnekliği

Bloklar iki farklı normalizasyonu destekler:

- **BatchNorm** → Standart CNN eğitimi
- **GroupNorm** → Küçük batch boyutlarında stabilite

Bu, modeli farklı donanım ve veri senaryolarına uyumlu hale getirir.

---

## ✨ Attention Entegrasyonu

Blok, residual toplamadan önce **attention modülü eklenmesine izin verir**. Böylece:

- Residual öğrenme korunur
- Kanal veya uzamsal ağırlıklandırma yapılabilir

Bu, performansı artıran modern bir genişletmedir.

---

## 🏗 MiniResNet Yapısı

Model dört stage’den oluşur:

| Stage   | Kanal Artışı | Çözünürlük  |
| ------- | ------------ | ----------- |
| Stage 1 | Sabit        | Aynı        |
| Stage 2 | ×2           | Yarıya iner |
| Stage 3 | ×2           | Yarıya iner |
| Stage 4 | ×2           | Yarıya iner |

Her stage birden fazla residual blok içerir.

---

## 💡 Bu Yaklaşımın Avantajları

✔ Derin ağlar stabil eğitilir  
✔ Gradyan akışı korunur  
✔ Parametre verimliliği yüksektir  
✔ Attention ve farklı norm türleriyle genişletilebilir  
✔ Görüntü sınıflandırma backbone’u olarak güçlüdür

---

## 🔚 Özet

Bu modelde uygulanan residual yaklaşım:

**Temiz, klasik Basic Residual tasarımı + esnek norm + opsiyonel attention + stage tabanlı derinleştirme**

Modern CNN backbone’larının temel yapı taşlarından biridir.
