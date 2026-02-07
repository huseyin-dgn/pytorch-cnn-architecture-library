# Controlled Attention–Residual Fusion (Pattern-2)

Bu yapı, klasik **Attention–Residual fusion**’ın daha kontrollü bir versiyonudur.  
Amaç: Attention’ın residual temsili **ne kadar etkileyeceğini öğrenilebilir bir katsayıyla ayarlamak**.

---

## 🎯 Temel Denklem

Önce klasik residual çıktı üretilir:

**z = Skip(x) + F(x)**

Ardından attention uygulanmış versiyon elde edilir:

**z_att = A(z) ⊙ z**

Son olarak iki temsil karıştırılır:

**Çıkış = Aktivasyon( (1−λ)·z + λ·z_att )**

---

## 🧠 Pattern’in Mantığı

Bu tasarımda attention doğrudan zorla uygulanmaz.  
Bunun yerine model şuna karar verir:

> “Residual temsil mi daha önemli, yoksa attention ile filtrelenmiş temsil mi?”

Bu dengeyi **λ (lambda)** belirler.

---

## 🔀 Pattern-2 Fusion Özelliği

Bu pattern, klasik “attention → residual toplama” yerine:

**Önce residual oluştur → sonra attention ile yeniden ağırlıklandır → iki temsili karıştır**

şeklinde çalışır.

Bu yüzden adı:

> **Post-Residual Controlled Attention Fusion**

---

## ⚙️ λ (Lambda) Nedir?

- 0’a yakın → Model daha çok saf residual’a güvenir  
- 1’e yakın → Model attention filtreli temsile daha çok güvenir  
- Öğrenilebilir ise → Eğitim sırasında en iyi dengeyi bulur  

Bu, attention’ın aşırı baskın olup öğrenmeyi bozmasını engeller.

---

## 🧩 Attention Türü

CBAM kullanılır:

| Tür | Ne Seçer |
|-----|----------|
| Channel Attention | Hangi feature kanalları önemli |
| Spatial Attention | Hangi uzamsal bölgeler önemli |

Ama bu attention artık **zorunlu değil**, λ ile kontrollü.

---

## 🚀 Neden Bu Pattern Güçlü?

✔ Attention’ın aşırı agresif etkisi kontrol edilir  
✔ Residual stabilite korunur  
✔ Model, hangi temsile güveneceğini öğrenir  
✔ Gürültülü veri senaryolarında daha dengeli çalışır  

---

## 🔚 Özet

Bu pattern:

**Residual öğrenme + Attention + Öğrenilebilir karışım katsayısı**

yapısını kullanır ve klasik attention-residual fusion’dan daha esnek ve stabil bir tasarım sunar.
