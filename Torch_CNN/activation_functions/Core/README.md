## ✅ Aktivasyon Fonksiyonları – Kesin Sonuç (15 Epoch)

### 📊 Final Epoch (15/15) Sonuçları

| Activation | Final Train Loss | Final Test Loss | Final Test Accuracy |
| ---------- | ---------------- | --------------- | ------------------- |
| ReLU       | 0.4770           | 0.6740          | 0.7797              |
| SiLU       | 0.4712           | 0.5892          | 0.8029              |
| LeakyReLU  | 0.5041           | 0.5645          | 0.8088              |
| **PReLU**  | **0.4614**       | **0.5573**      | **0.8149**          |

---

### 🏆 En İyi Değerler (Tüm Eğitim Boyunca)

| Activation | Peak Accuracy | Peak Epoch   |
| ---------- | ------------- | ------------ |
| ReLU       | 0.7915        | Epoch 14     |
| SiLU       | 0.8102        | Epoch 14     |
| LeakyReLU  | 0.8088        | Epoch 15     |
| **PReLU**  | **0.8149**    | **Epoch 15** |

---

## 🧠 Kesin Yorum ve Karar

- **PReLU açık ara en iyi aktivasyon**:
  - En yüksek **final accuracy**
  - En düşük **test loss**
  - Eğitim sonunda performans **düşmüyor**, aksine güçleniyor
- **LeakyReLU**, parametresiz olmasına rağmen oldukça güçlü ve stabil:
  - Deploy / sade mimari için iyi alternatif
- **SiLU**, modern ve güçlü olsa da:
  - Bu deneyde **PReLU ve LeakyReLU’nun gerisinde**
- **ReLU**, en zayıf genel performansı gösteriyor:
  - Finalde hem accuracy düşüyor hem test loss yükseliyor

---

## 🔒 Kesin Sonuç (Tek Cümle)

> **Bu deney koşullarında PReLU, hem doğruluk hem genelleme açısından en iyi aktivasyon fonksiyonudur; LeakyReLU ikinci en iyi ve deploy açısından en dengeli alternatiftir.**

---

## 📌 Proje / Repo için Net Kullanım Kararı

- **Default activation:** `PReLU`
- **Lightweight / deploy alternatifi:** `LeakyReLU`
- **ReLU:** ⚠️ opsiyonel
- **SiLU:** ⚠️ opsiyonel ama birincil tercih değil
