## Dynamic Convolution (CondConv/DyConv tarzı)

Dynamic Conv’un olayı: **tek bir sabit kernel yerine, input’a göre kernel seçmek/karıştırmak.**  
Yani her örnek (batch içindeki her görüntü) için konvolüsyon ağırlığı farklı oluyor. 🧠⚙️

### Mantık (net)

- Elinde **K tane kernel** var (kernel bankası)
- Bir **router (gating)** ağı input’tan bakıp `a = [a1..aK]` ağırlıklarını üretir
- Gerçek kullanılan kernel:
  - `W_dyn = Σ_k a_k * W_k`
- Böylece model “bu görüntüde şu filtre kombinasyonu daha iyi” diye kendisi karar verir ✅

---

## Bu kodda ne oluyor?

### 1) RoutingMLP (Gating)

- `GAP` ile `(B,C,H,W) -> (B,C)` özet çıkarıyor
- 2 katmanlı MLP ile `(B,K)` logits üretiyor
- `softmax(logits / temperature)` ile **ağırlıklar**:
  - her satır toplamı = 1
  - yani bir **karışım oranı** (mixture weights)

`temperature`:

- düşük → daha “sert seçim” (one-hot’a yaklaşır)
- yüksek → daha “yumuşak karışım”

### 2) DynamicConv2d

- Kernel bankası: `weight` şekli `(K, cout, cin, k, k)`
- Router’dan gelen `a (B,K)` ile batch başına kernel üretiliyor:
  - `Wdyn = einsum("bk,kocij->bocij", a, weight)` → `(B, cout, cin, k, k)`
- Sonra **her örnek için ayrı conv** uygulanıyor (for-loop):
  - çünkü her sample’ın ağırlığı farklı ✅

> Not: Bu “okunur demo” yolu. Performans için genelde group-conv hilesiyle vektörize edilir.

### 3) DynamicBlock

- `DynamicConv2d -> BN -> ReLU`
- İstersen routing’i de dışarı veriyor (debug için güzel)

### 4) DynamicCNN

- stem normal conv
- 3 tane dynamic stage (stride=2 ile downsample)
- `return_routing=True` olunca her stage’in `a` vektörlerini döndürüyor

---

## Ne işe yarar?

- Aynı ağ içinde **çoklu uzman (mixture of experts)** gibi davranır 🎛️
- Görüntünün içeriğine göre filtre kombinasyonunu değiştirir
- Özellikle çeşitliliği yüksek veri dağılımlarında fayda görebilirsin

---

## Dikkat (gerçekçi uyarı)

- Bu implementasyonda batch içi for-loop var → büyük batch’te yavaşlar ⚠️
- Routing yanlış ayarlanırsa (çok yüksek/çok düşük temperature) öğrenme zorlaşabilir.
- K büyüdükçe kapasite artar ama maliyet de artar.

---
