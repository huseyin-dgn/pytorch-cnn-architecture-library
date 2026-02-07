# Pre-Activation Residual Yapı (ResNet v2) — Tek Yapı Özeti

## 🎯 Amaç
Çok derin ağlarda **gradyan akışını en temiz hâle getirmek** ve eğitimi daha stabil yapmak.

---

## 🧠 Yapının Mantığı

Bu tasarımda kritik fark şudur:

> **Normalizasyon ve aktivasyon, konvolüsyondan ÖNCE uygulanır.**

Blok şu akışı izler:

**Norm → Aktivasyon → Conv → Norm → Aktivasyon → Conv → Skip ile toplama**

---

## 🔁 Residual Toplama

- Giriş (x) doğrudan kısa yoldan taşınır.  
- Ana yol F(x) öğrenilen dönüşümdür.  
- Çıkış: **x + F(x)**  
- Toplama sonrası genellikle ek aktivasyon uygulanmaz.

---

## 📌 Neden Bu Sıra Kullanılır?

✔ Skip yolu tamamen lineer kalır  
✔ Gradyan, aktivasyon bariyerine takılmaz  
✔ Çok derin ağlar daha kolay eğitilir  
✔ Degradation problemi azalır  

---

## ⚙️ Skip Yol Davranışı

| Durum | İşlem |
|------|-------|
| Boyutlar aynı | Giriş direkt eklenir |
| Kanal/çözünürlük farklı | 1×1 projeksiyonla eşitlenir |

---

## 🔬 Post-Act ile Farkı

| | Post-Activation | Pre-Activation |
|--|----------------|----------------|
| Aktivasyon yeri | Conv sonrası | Conv öncesi |
| Toplama sonrası ReLU | Var | Yok (genelde) |
| Gradyan akışı | Daha sınırlı | Daha temiz |
| Derin ağ uyumu | Orta | Çok yüksek |

---

## 🔚 Özet

Pre-activation residual yapı:

**Aktivasyonu öne alır, residual hattı lineer tutar ve derin CNN’lerde maksimum eğitim stabilitesi sağlar.**
