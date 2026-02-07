# ResNeXt Grouped Residual Blok

Bu yapı, klasik bottleneck residual bloğun **ResNeXt versiyonudur**. Temel fark, orta konvolüsyonun **grouped (cardinality tabanlı)** yapılmasıdır. Amaç, parametre maliyetini çok artırmadan model kapasitesini yükseltmektir.

---

## 🎯 Temel Denklem

**Çıkış = Aktivasyon( F(x) + Skip(x) )**

Residual mantık aynıdır; değişen şey **F(x)**’in iç yapısıdır.

---

## 🧠 F(x) Yapısı (Bottleneck + Grouped Conv)

Ana yol üç adımdan oluşur:

1. **1×1 Conv (kanal ayarı)**  
   Girişi grouped 3×3 için uygun genişliğe getirir.

2. **3×3 Grouped Conv**  
   Özellik çıkarımı burada yapılır ama kanallar **gruplara bölünerek** işlenir.

3. **1×1 Conv (genişletme)**  
   Çıkış kanalı, klasik bottleneck expansion ile büyütülür.

---

## 🔢 Cardinality (Gruplar) Nedir?

ResNeXt’in ana gücü buradan gelir:

| Parametre      | Anlamı                  |
| -------------- | ----------------------- |
| **groups**     | Kaç paralel grup olduğu |
| **base_width** | Her grubun genişliği    |

Gruplar arttıkça model:

✔ Daha fazla paralel özellik yolu öğrenir  
✔ Daha zengin temsil üretir  
✔ Derinliği artırmadan kapasite büyütür

Bu, “genişlik” yerine **çoklu yol çeşitliliği** sağlar.

---

## 🔁 Skip (Kısa Yol)

Standart residual kuralları geçerli:

| Durum               | İşlem           |
| ------------------- | --------------- |
| Boyutlar aynı       | Kimlik geçişi   |
| Kanal/stride farklı | 1×1 projeksiyon |

---

## 🧩 Klasik Bottleneck’ten Farkı

|                  | ResNet Bottleneck | ResNeXt Bottleneck  |
| ---------------- | ----------------- | ------------------- |
| Orta katman      | Normal 3×3        | Grouped 3×3         |
| Kapasite artışı  | Kanal genişletme  | Cardinality artırma |
| Parametre verimi | Orta              | Daha verimli        |

---

## 🚀 Neden Güçlü?

✔ Daha az parametreyle daha zengin temsil  
✔ Overfitting riski daha düşük  
✔ Derin ağlarda daha iyi genelleme  
✔ Modern detection ve classification backbone’larında yaygın

---

## 🔚 Özet

Bu residual yapı:

**Bottleneck tasarımı + Grouped convolution (cardinality)**  
ile klasik ResNet’ten daha güçlü ve verimli bir temsil öğrenme sağlar.
