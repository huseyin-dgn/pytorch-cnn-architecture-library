# Cutout

Bu dokümanda Cutout regularization tekniği açıklanmaktadır.

Cutout, veri artırma (data augmentation) tabanlı bir regularization yöntemidir ve özellikle görüntü sınıflandırma problemlerinde kullanılır.

---

# 1) Cutout Nedir?

Cutout, giriş görüntüsünün rastgele seçilen kare bir bölgesinin sıfırlanması (maskelenmesi) prensibiyle çalışır.

Yani:

- Görüntüden bir dikdörtgen alan seçilir.
- Bu alan tamamen siyah (veya sabit bir değer) yapılır.
- Etiket değişmez.

---

# 2) Nasıl Çalışır?

Verilen bir görüntü için:

1. Rastgele merkez noktası seçilir.
2. Belirli `mask_size` kadar kare alan belirlenir.
3. Bu alan sıfırlanır.
4. Görüntü model girişine bu haliyle verilir.

Önemli nokta:

- Bu işlem yalnızca training sırasında uygulanır.
- Test sırasında Cutout kapalıdır.

---

# 3) Temel Amaç

CNN’ler genellikle:

- Görüntünün en ayırt edici bölgesine aşırı bağımlı olur.
- Lokal feature’lara aşırı odaklanır.

Cutout:

- Modeli eksik bilgiyle öğrenmeye zorlar.
- Daha geniş bağlamsal özellikleri öğrenmesini sağlar.
- Overfitting’i azaltır.

---

# 4) Matematiksel Sezgi

Cutout bir maske işlemi olarak düşünülebilir:

x' = x ⊙ M

Burada:

- M = 0/1 mask
- Maskenin belirli bölgesi 0’dır
- Etiket aynı kalır

Bu yöntem label karıştırmaz, yalnızca görüntü manipülasyonu yapar.

---

# 5) Nerede Kullanılır?

- CIFAR-10 / CIFAR-100
- Küçük görüntü datasetleri
- Classification görevleri

Detection tarafında dikkatli kullanılmalıdır çünkü:

- Bounding box alanı silinebilir
- Küçük objeler tamamen kaybolabilir

---

# 6) Cutout vs Diğer Teknikler

| Teknik    | Giriş Seviyesi | Label Değişir | Amaç                               |
| --------- | -------------- | ------------- | ---------------------------------- |
| Dropout   | Feature map    | Hayır         | Nöron regularization               |
| DropBlock | Feature map    | Hayır         | Spatial regularization             |
| Cutout    | Input image    | Hayır         | Görüntü seviyesinde regularization |
| CutMix    | Input image    | Evet          | Görüntü + label karışımı           |

---

# 7) Avantajlar

- Basit
- Ek parametre gerektirmez
- Küçük datasetlerde etkili
- Modeli daha robust hale getirir

---

# 8) Dezavantajlar

- Küçük objeleri tamamen silebilir
- Detection için agresif olabilir
- Çok büyük mask size performansı düşürebilir

---

# 9) Ne Zaman Kullanılmalı?

- Küçük dataset
- Overfitting gözleniyorsa
- Classification problemi

Detection için genellikle:

- CutMix
- Mosaic
- Multi-scale training

daha yaygındır.

---

# Sonuç

Cutout:

Giriş görüntüsünün rastgele bir kısmını silerek modeli eksik bilgiyle öğrenmeye zorlayan bir regularization yöntemidir.

Classification problemlerinde etkilidir.
Detection görevlerinde dikkatli kullanılmalıdır.
