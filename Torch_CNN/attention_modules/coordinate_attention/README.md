# Coordinate Attention (CA)

## Amaç

Coordinate Attention, kanal dikkatini **uzamsal konum bilgisiyle** birleştirir.  
Hem kanal seçimi yapar hem de uzamsal yön bilgisi taşır.

---

## Temel Fikir

Geleneksel attention mekanizmaları uzamsal bilgiyi tamamen sıkıştırır.  
CA ise:

> Yatay ve dikey yönlerde ayrı ayrı global bilgi toplar

Bu sayede konum bilgisi kaybolmaz.

---

## Nasıl Çalışır?

1. Feature map iki yönde ayrı ayrı sıkıştırılır
2. Kanal etkileşimi öğrenilir
3. Konum bilgisi korunarak dikkat haritası üretilir

---

## Avantajları

✔ Uzamsal yön bilgisi korunur  
✔ Küçük nesnelerde daha etkili olabilir  
✔ Hem channel hem spatial özellik taşır

---

## Dezavantajları

✘ SE/ECA’dan daha karmaşık  
✘ Ek hesaplama yükü getirir

---

## Ne Zaman Kullanılır?

- Detection görevlerinde
- Küçük nesne içeren veri setlerinde
- Konum hassasiyeti önemli olduğunda
