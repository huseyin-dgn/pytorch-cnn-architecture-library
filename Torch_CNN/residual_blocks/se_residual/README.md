# SE (Squeeze-and-Excitation) Residual Blok

Bu yapı, klasik residual bloğa **kanal dikkati (channel attention)** ekler. Amaç, ağın “hangi feature kanalları daha önemli?” sorusunu öğrenmesini sağlamaktır.

---

## 🎯 Temel Denklem

**Çıkış = Aktivasyon( Skip(x) + SE(F(x)) )**

Burada SE modülü, residual kolun ürettiği özellikleri kanal bazında yeniden ağırlıklandırır.

---

## 🧠 SE Mekanizması Nasıl Çalışır?

SE iki adımdan oluşur:

### 1️⃣ Squeeze  
Her kanal için uzamsal bilgi özetlenir (global average pooling).

→ Model, her kanalın genel önemini ölçer.

### 2️⃣ Excitation  
Küçük bir MLP ile kanal başına ağırlık katsayıları üretilir.

→ Hangi kanallar güçlendirilecek, hangileri bastırılacak öğrenilir.

Sonuçta bir **kanal maskesi** elde edilir ve residual özellikler bu maske ile çarpılır.

---

## 🔁 Residual Yapı Korunur

SE yalnızca **F(x)** üzerine uygulanır:

✔ Skip yolu aynen kalır  
✔ Toplama işlemi değişmez  
✔ Gradyan akışı residual hat üzerinden stabil kalır  

Bu yüzden model stabilitesini bozmaz.

---

## 🧩 Ne Kazandırır?

✔ Önemli kanallar güçlenir  
✔ Gürültülü kanallar bastırılır  
✔ Feature seçiciliği artar  
✔ Parametre artışı çok düşüktür  

---

## 🆚 Klasik Residual’dan Farkı

| | Normal Residual | SE Residual |
|--|----------------|-------------|
| Kanal önemi | Sabit | Öğrenilebilir |
| Attention | Yok | Kanal bazlı |
| Maliyet artışı | — | Çok düşük |

---

## 🚀 Nerede Etkilidir?

- Sınıflandırma backbone’larında  
- Detection ve segmentation modellerinde  
- Derin katmanlarda semantik özelliklerin ayrımında

---

## 🔚 Özet

SE residual yapı:

**Residual öğrenme + Kanal bazlı dikkat**

kombinasyonudur ve modeli daha seçici ve güçlü hale getirir.
