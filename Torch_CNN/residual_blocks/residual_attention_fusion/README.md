# Attention–Residual Fusion Pattern — Kısa Özet

## 🎯 Amaç
Residual öğrenmeyi korurken, ağın **önemli özelliklere odaklanmasını** sağlamak.

---

## 🧠 Temel Fikir

Residual blokta ana yol **F(x)** öğrenir, skip yol ise bilgiyi taşır.  
Attention modülü, **F(x)’in ürettiği özellikleri ağırlıklandırır**.

**Çıkış = Aktivasyon( Attention(F(x)) + Skip(x) )**

Yani:
- Residual yapı korunur  
- Attention, “neyin önemli olduğunu” söyler

---

## 🔀 Fusion (Birleşim) Noktası

Attention genelde şu noktada uygulanır:

**Conv → Norm → Conv → Norm → Attention → Toplama**

Yani toplama öncesi ana yol filtrelenir.

---

## 🧩 Ne Kazandırır?

✔ Gürültülü feature’lar bastırılır  
✔ Önemli kanallar/bölgeler öne çıkar  
✔ Residual stabilite bozulmaz  
✔ Özellikle detection ve segmentation’da etkilidir  

---

## ⚙️ Uygulama Türleri

| Attention Türü | Ne Ağırlıklandırır |
|---------------|--------------------|
| Channel Attention | Hangi kanal önemli |
| Spatial Attention | Hangi bölge önemli |
| SE / CBAM / ECA | Hafif ve etkili modüller |

---

## 🔚 Özet

Bu pattern:

**Residual öğrenme + Attention odaklanması = Daha seçici ve güçlü feature temsili**
