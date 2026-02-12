# Spatial Dropout (Dropout2d)

Bu modülde **Spatial Dropout (Channel-wise Dropout)** implement edilmiştir.

Amaç: CNN'lerde kanal bağımlılığını (co-adaptation) azaltmak ve daha robust feature öğrenimi sağlamak.

---

## 📌 Ne Yaptık?

- Klasik `Dropout` (element-wise) ile karşılaştırma yaptık.
- CIFAR-100 üzerinde **aynı mimari** ile deney gerçekleştirdik.
- `nn.Dropout2d` kullanarak kanal bazlı maske uyguladık.
- Dropout ve Spatial Dropout performanslarını:
  - train_loss
  - train_acc
  - val_loss
  - val_acc
    üzerinden kıyasladık.
- Sonuçları grafikle görselleştirdik.
- Underfit / overfit analizini yaptık.

---

## 🔬 Teknik Detay

Girdi tensörü:  
`[B, C, H, W]`

Maske şekli:  
`[B, C, 1, 1]`

Eğitim sırasında: **y = x \* mask / (1 - p)**

- Kanal ya tamamen tutulur
- Ya tamamen sıfırlanır
- Eval modunda dropout kapalıdır

---

## 🧠 Kullanım Yeri

Spatial Dropout, residual branch veya Conv-BN-Act sonrası uygulanır: **Conv → BN → Act → Dropout2d**

Önerilen kullanım:

- Erken katmanlarda düşük oran
- Derin katmanlarda daha etkili
- p genelde: 0.05 – 0.15

---

## 🎯 Sonuç

Bu deneyde:

- CIFAR-100 üzerinde klasik Dropout daha iyi performans verdi.
- Spatial Dropout, yanlış yerleşim veya yüksek p değerinde underfit'e sebep olabilir.

Doğru oran ve doğru konumlandırma ile tekrar test edilmelidir.
