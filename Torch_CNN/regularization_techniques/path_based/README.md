# DropPath ve Stochastic Depth

Bu dokümanda iki benzer fakat kavramsal olarak ayrılması gereken regularization tekniği açıklanmaktadır:

- Stochastic Depth
- DropPath

Her iki teknik de özellikle **residual mimarilerde** kullanılır ve temel amaçları
derin ağların genelleme kabiliyetini artırmaktır.

---

# 1) Stochastic Depth

## Tanım

Stochastic Depth, residual blokların içindeki ana dönüşüm yolunun
eğitim sırasında rastgele kapatılması prensibiyle çalışan bir tekniktir.

Yani:

out = x + F(x)

ifadesinde, eğitim sırasında bazen:

out = x

şeklinde davranılır.

## Nasıl Çalışır?

- Her residual blok için bir drop olasılığı belirlenir.
- Eğitim sırasında, bazı örneklerde residual branch tamamen kapatılır.
- Shortcut (identity) her zaman açık kalır.
- Test sırasında tüm bloklar aktif olur.

## Amaç

- Ağı etkili olarak daha sığ hale getirmek.
- Gradient akışını stabilize etmek.
- Çok derin ağlarda overfitting’i azaltmak.

## Önemli Nokta

Bu teknik sadece residual mimarilerde anlamlıdır.

---

# 2) DropPath

## Tanım

DropPath, Stochastic Depth’in daha genelleştirilmiş versiyonudur.

Residual branch’in tamamını ya açık ya kapalı yapar.
Ancak bunu daha esnek şekilde uygular:

- Batch bazlı
- Schedule ile artan oranlı
- Derinliğe bağlı ölçekleme ile

## Nasıl Çalışır?

Residual çıkışı:

x \* mask / keep_prob

şeklinde ölçeklenir.

Mask:

- 1 → branch aktif
- 0 → branch kapalı

Expected value korunur (inverted scaling).

## Ek Özellikler

DropPath genellikle şunları içerir:

- Layer-wise scaling (derin katmanlarda daha yüksek drop)
- Warmup schedule
- Batchwise veya elementwise opsiyon

---

# 3) Temel Fark

| Özellik          | Stochastic Depth | DropPath          |
| ---------------- | ---------------- | ----------------- |
| Çalışma alanı    | Residual branch  | Residual branch   |
| Granülerlik      | Sample-wise      | Genelleştirilmiş  |
| Schedule desteği | Genelde sabit    | Var               |
| Layer scaling    | Genelde manuel   | Yerleşik olabilir |
| Modern kullanım  | Var              | Daha yaygın       |

---

# 4) Neden Çalışır?

Derin ağlarda:

- Her blok öğrenmeye katkı verir
- Ama bazı bloklar aşırı uyum sağlar

DropPath / Stochastic Depth:

- Rastgele bazı blokları kapatarak
- Ensemble benzeri etki oluşturur
- Gradient varyansını düzenler
- Overfitting’i azaltır

---

# 5) Nerede Kullanılır?

- ResNet
- EfficientNet
- Vision Transformer (çok yaygın)
- Modern CNN backbone’lar

---

# 6) Mühendislik Perspektifi

Classification’da:

- Hafif bir performans artışı sağlayabilir.

Detection backbone’unda:

- Derin yapıda genelleme artırır.

ViT tarafında:

- Standart regularization haline gelmiştir.

---

# 7) Ne Zaman Kullanmalı?

- Çok derin model varsa
- Overfitting gözleniyorsa
- Residual mimari kullanılıyorsa

---

# Sonuç

Stochastic Depth:

- Orijinal konsepttir.
- Residual branch’i rastgele kapatır.

DropPath:

- Daha esnek ve modern implementasyondur.
- Schedule ve layer scaling desteği sunar.

Pratikte çoğu modern projede DropPath tercih edilir.
