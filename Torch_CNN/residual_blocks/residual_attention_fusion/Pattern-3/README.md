# Controlled Pre-Attention Residual Fusion (Pattern-3)

Bu yapı, attention ile residual öğrenmeyi **daha erken bir aşamada** birleştirir.  
Önce giriş özelliği attention ile yumuşak şekilde yeniden ağırlıklandırılır, sonra residual dönüşüme sokulur.

---

## 🎯 Temel Denklem

Önce giriş için iki temsil oluşturulur:

- **x** → ham özellik
- **x_att = A(x) ⊙ x** → attention ile filtrelenmiş giriş

Bunlar karıştırılır:

**x̃ = (1−λ)·x + λ·x_att**

Sonra residual blok uygulanır:

**Çıkış = Aktivasyon( Skip(x) + F(x̃) )**

---

## 🧠 Pattern’in Farkı

Klasik attention-residual pattern’de attention, **F(x)** üzerine uygulanır.  
Burada ise attention:

> **Residual öğrenme başlamadan ÖNCE giriş temsiline uygulanır.**

Yani attention, dönüşümün girdisini değiştirir.

---

## 🔀 Fusion Tipi

Bu pattern:

**Pre-Residual Controlled Attention Fusion**

olarak düşünülebilir.

- Attention sonrası temsil, residual branch’e girer
- Skip yolu değişmez
- Toplama yine residual mantıkta yapılır

---

## ⚙️ λ (Lambda) Rolü

λ, modelin şuna karar vermesini sağlar:

| λ değeri      | Davranış                   |
| ------------- | -------------------------- |
| Küçük         | Ham feature’lar korunur    |
| Büyük         | Attention etkisi artar     |
| Öğrenilebilir | Model en iyi dengeyi bulur |

Bu, attention’ın aşırı baskın olmasını engeller.

---

## 🧩 Neden Bu Yapı İlginç?

✔ Attention, residual öğrenmenin girişini yönlendirir  
✔ Gürültü daha erken bastırılır  
✔ Skip hattı yine saf kalır  
✔ Residual stabilite korunur  
✔ Feature refinement daha “ön aşamada” yapılır

---

## 🆚 Pattern-2 ile Farkı

|                   | Pattern-2          | Pattern-3          |
| ----------------- | ------------------ | ------------------ |
| Attention nerede? | F(x) sonrası       | F(x)’ten önce      |
| Karıştırılan şey  | Residual çıktılar  | Giriş feature’ları |
| Etki alanı        | Temsil iyileştirme | Temsil yönlendirme |

---

## 🔚 Özet

Bu pattern:

**Attention ile yönlendirilmiş giriş + Residual öğrenme + Kontrollü karışım**

yaklaşımıdır ve attention’ı residual dönüşümün daha erken aşamasına taşır.
