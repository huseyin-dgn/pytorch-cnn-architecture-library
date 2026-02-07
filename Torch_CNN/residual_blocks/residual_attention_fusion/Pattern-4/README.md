# Dual Near-Identity Gated Residual Fusion (Pattern-4)

Bu yapı, residual bloğun **iki kolunu da (skip ve residual)** attention ile kontrol eder.  
Ama dikkat: Attention doğrudan çarpıp bozmaz; **kimliğe yakın (near-identity) kapılar** ile yumuşak şekilde etki eder.

---

## 🎯 Temel Denklem

Önce iki yol oluşturulur:

- **Skip yolu:** `identity = Skip(x)`
- **Residual yolu:** `f = F(x)`

Her iki yol da ayrı ayrı kapılanır:

- **s = identity ⊙ (1 + γₛ · Aₛ(identity))**
- **r = f ⊙ (1 + γᵣ · Aᵣ(f))**

Sonra:

**Çıkış = Aktivasyon( s + r )**

---

## 🧠 Pattern’in Farkı

Klasik attention-residual yapılarında attention sadece residual kola uygulanır.  
Burada ise:

> **Skip yolu bile hafifçe modüle edilir.**

Ama “near-identity” olduğu için temel bilgi akışı bozulmaz.

---

## 🔒 Near-Identity Gate Mantığı

Kapı formu:

**x → x · (1 + γ·mask(x))**

Bu şunları sağlar:

| Özellik | Sonuç                           |
| ------- | ------------------------------- |
| γ ≈ 0   | Neredeyse saf residual davranış |
| γ küçük | Hafif feature vurgusu           |
| γ büyük | Daha güçlü dikkat etkisi        |

Yani attention etkisi **güvenli ve kontrollü**.

---

## 🔀 Çift Attention (Dual Mask)

| Kol                    | Amaç                                  |
| ---------------------- | ------------------------------------- |
| **Aₛ** (Skip mask)     | Taşınan temel bilgiyi hafifçe ayarlar |
| **Aᵣ** (Residual mask) | Öğrenilen dönüşümü seçici güçlendirir |

Bu sayede hem temel sinyal hem öğrenilen özellik filtrelenir.

---

## ⚙️ Neden Güçlü Bir Tasarım?

✔ Skip hattı tamamen “pasif” değil, ama bozulmuyor  
✔ Residual özellikler seçici biçimde güçleniyor  
✔ Attention etkisi aşırıya kaçmıyor  
✔ Derin katmanlarda semantik özellikler daha iyi vurgulanıyor

---

## 🧩 Klasik Residual’dan Farkı

|                     | Normal Residual  | Pattern-4                 |
| ------------------- | ---------------- | ------------------------- |
| Skip yolu           | Sabit            | Hafif attention kontrollü |
| Residual yolu       | Doğrudan eklenir | Attention ile kapılanır   |
| Kontrol parametresi | Yok              | γ (öğrenilebilir)         |

---

## 🚀 Nerede Anlamlı?

- Orta ve derin stage’lerde
- Semantik özelliklerin ağır bastığı katmanlarda
- Gürültülü veri veya karmaşık sahnelerde

---

## 🔚 Özet

Pattern-4:

**Residual öğrenme + Skip ve Residual için ayrı attention + Near-identity kapılar**

yaklaşımıdır. Hem stabil residual akışını korur hem de feature’ları akıllı şekilde yeniden ağırlıklandırır.
