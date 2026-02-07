## Deformable Convolution (DCN) — Kısa Açıklama

Bu blokta olay şu: **konvolüsyonun örnekleme noktaları sabit değil, öğrenilerek kaydırılıyor.**  
Klasik conv her yerde aynı 3×3 grid’i kullanır. Deformable conv ise her konum için o grid’i **(dx, dy)** offset’leriyle esnetir.

### Bu kodda ne oluyor?

- **offset_conv**: Giriş feature map’ten offset üretir.  
  Kanal sayısı: `2 * K * K`  
  (Her kernel noktası için dx ve dy var)
- **DeformConv2d**: Üretilen offset ile sampling noktalarını kaydırıp conv uygular.
- Sonra **BatchNorm + ReLU** ile stabilize eder.

### Ne işe yarar?

- Poz değişimi, perspektif, deformasyon gibi durumlarda
  **klasik conv’a göre daha uyumlu feature** çıkarabilir.
- Özellikle detection / segmentation gibi işlerde işe yarar.

### Bedeli var mı?

Var: Offset üretimi + deformable sampling → **daha maliyetli** (yavaş/çok compute).  
O yüzden genelde her katmana koymak yerine orta katmanlarda seçmeli kullanılır.
