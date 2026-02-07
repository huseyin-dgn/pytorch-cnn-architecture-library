# Wide Residual Blok (Wide ResNet Yaklaşımı)

Bu yapı, klasik residual mimarinin **derinleştirilmesi yerine genişletilmesi** fikrine dayanır. Amaç: Ağı aşırı derin yapmak yerine kanalları artırarak daha güçlü temsil öğrenmek.

---

## 🎯 Temel Denklem

**Çıkış = F(x) + Skip(x)**

Residual mantık değişmez; fark, blokların **kanal sayısının büyütülmüş olmasıdır**.

---

## 🧠 Blok Yapısı

Bu blok **pre-activation residual** düzenini kullanır:

**BN → ReLU → Conv → (Dropout) → BN → ReLU → Conv**

Toplama en sonda yapılır. Aktivasyon toplama sonrası ayrı uygulanmaz.

Bu sıralama:

✔ Gradyan akışını iyileştirir  
✔ Çok katmanlı geniş ağlarda stabilite sağlar

---

## 🔀 “Wide” Ne Demek?

| Parametre            | Anlamı                               |
| -------------------- | ------------------------------------ |
| **widen_factor (k)** | Tüm stage kanallarını çarpan katsayı |
| **depth = 6n+4**     | Kaç residual blok olduğu             |

Kanal artışı:

- Stage 1: 16 × k
- Stage 2: 32 × k
- Stage 3: 64 × k

Yani ağ derinleşmez, **her katman daha geniş temsil kapasitesine sahip olur**.

---

## 🔁 Skip Yolu

Standart residual kuralı geçerli:

| Durum                       | İşlem           |
| --------------------------- | --------------- |
| Boyutlar aynı               | Identity        |
| Kanal veya stride değişiyor | 1×1 projeksiyon |

---

## 💡 Dropout Neden Var?

Wide ResNet’te genişlik arttıkça overfitting riski artar.  
Blok ortasına konan dropout:

✔ Regularization sağlar  
✔ Geniş ağın aşırı ezberlemesini önler

---

## 🆚 Klasik ResNet ile Farkı

|               | ResNet       | Wide ResNet         |
| ------------- | ------------ | ------------------- |
| Derinlik      | Yüksek       | Orta                |
| Kanal sayısı  | Daha dar     | Daha geniş          |
| Öğrenme stili | Derin temsil | Geniş temsil        |
| Performans    | İyi          | Çoğu zaman daha iyi |

---

## 🚀 Neden Etkili?

✔ Daha iyi gradyan akışı  
✔ Daha az katmanla yüksek kapasite  
✔ Eğitimi daha hızlı  
✔ CIFAR tarzı veri setlerinde çok güçlü sonuçlar

---

## 🔚 Özet

Wide residual yapı:

**Pre-activation residual tasarım + kanal genişletme (k faktörü)**  
kullanarak derinliği artırmadan model kapasitesini yükseltir.
