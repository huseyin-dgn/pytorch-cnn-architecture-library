# Bottleneck Residual Blok (ResNet-50/101 Tarzı)

## 🎯 Amaç

- Derin ağlarda **gradyan kaybını önlemek**
- Daha derin mimari kurarken **eğitilebilirliği korumak**
- Hesap maliyetini kontrol altında tutmak

---

## 🧠 Temel Fikir

Bir blok iki yolu toplar:

**Çıkış = Aktivasyon( F(x) + Skip(x) )**

- **F(x)** → Öğrenen ana yol
- **Skip(x)** → Girişi direkt taşıyan kısa yol

Bu toplama, bilginin ve gradyanın katmanlar arasında kaybolmadan akmasını sağlar.

---

## 🧩 Bottleneck Mantığı

Ana yol üç adımdan oluşur:

1. **1×1 Daraltma**  
   Kanal sayısı düşürülür → hesap yükü azalır.

2. **3×3 İşlem**  
   Asıl uzamsal özellik çıkarımı burada yapılır.

3. **1×1 Genişletme**  
   Kanal sayısı tekrar artırılır → güçlü temsil kapasitesi.

Bu yapı sayesinde 3×3 konvolüsyon daha az kanalda çalışır.

---

## 🔁 Skip (Kısa Yol)

İki durum vardır:

| Durum                           | Skip Davranışı                |
| ------------------------------- | ----------------------------- |
| Kanal ve çözünürlük aynı        | Giriş direkt eklenir          |
| Kanal veya çözünürlük değişiyor | 1×1 projeksiyon ile eşitlenir |

---

## ⬇️ Downsample Nasıl Olur?

Çözünürlük düşürme gerekiyorsa:

- Ana yolda stride uygulanır
- Skip yol da aynı stride ile eşitlenir

Böylece toplama için boyutlar tutarlı olur.

---

## 💡 Neden Bu Yapı Kullanılır?

- Çok derin ağlar **stabil eğitilir**
- Özellik kaybı azalır
- Daha az maliyetle daha güçlü temsil elde edilir
- Detection ve backbone mimarilerinin temelidir

---

## 🏗 Özetle

Bu residual yapı:

✔ Bilgiyi katmanlar arasında taşır  
✔ Gradyan akışını korur  
✔ Hesap yükünü azaltır  
✔ Derin CNN'lerin temel yapı taşıdır
