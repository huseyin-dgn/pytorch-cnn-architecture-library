# Efficient Channel Attention (ECA)

## Amaç

ECA, kanal dikkatini (channel attention) **hafif ve parametresiz** bir şekilde üretmeyi amaçlar.  
SE bloklarının kanal ilişkisini öğrenme fikrini korur ancak MLP yerine daha basit bir mekanizma kullanır.

---

## Temel Fikir

SE bloğunda kanal etkileşimi iki tam bağlı katmanla modellenirken, ECA bu işlemi:

> **Yerel kanal etkileşimi (local cross-channel interaction)**

mantığı ile çözer.

Bu sayede:

- Parametre sayısı azalır
- Hesaplama maliyeti düşer
- Kanal bağımlılığı yine öğrenilebilir

---

## Nasıl Çalışır?

1. Global Average Pooling ile kanal özetleri çıkarılır
2. Bu vektör üzerinde küçük bir 1D konvolüsyon uygulanır
3. Çıkan değerler Sigmoid ile ağırlığa çevrilir
4. Kanallar yeniden ölçeklenir

---

## Avantajları

✔ Çok hafif  
✔ Ek parametre yükü çok az  
✔ Mobile / lightweight modeller için ideal

---

## Dezavantajları

✘ Kanal etkileşimi sınırlı bir komşulukta gerçekleşir  
✘ Çok karmaşık kanal ilişkilerini yakalamada SE kadar güçlü olmayabilir

---

## Ne Zaman Kullanılır?

- Hafif CNN mimarilerinde
- Mobil/edge senaryolarında
- Parametre bütçesi düşük olduğunda
