## Shift (ShiftConv) — Kısa Açıklama

ShiftConv’un olayı: **3×3 conv gibi “komşuluk (spatial) karışımı” yapmak ama bunu konvolüsyonla değil, kanalları kaydırarak yapmak.**  
Kaydırma **parametresizdir**. Asıl öğrenilebilir kısım sonradan gelen **1×1 conv**’dur. ⚙️

### Mantık (çekirdek fikir)

- 3×3 conv: komşu piksellerden ağırlıklı toplam alır → pahalı
- Shift: feature map’in bazı kanallarını
  - sağa, sola, yukarı, aşağı (ve istersen merkez) kaydırır
  - böylece “komşu bilgi” kanallar üzerinden taşınır ✅
- Sonra 1×1 conv ile bu kanallar karıştırılır → öğrenme burada olur ✅

### Bu implementasyonda ne yapılıyor?

#### 1) `shift2d(x, directions)`

- Giriş: `(B, C, H, W)`
- Kanalları `G = len(directions)` parçaya böler.
- Her parçayı farklı yöne `torch.roll` ile kaydırır.
- `mode="zero_pad"` seçilince wrap-around olan kısımlar sıfırlanır (padding gibi davranır).

Default yönler:

- `(0,0)` merkez + 4 komşu: sağ/sol/aşağı/yukarı

#### 2) `ShiftConvBlock`

Sıra net:

1. **shift2d** (parametre yok)
2. `stride!=1` ise **AvgPool ile downsample**
3. **1×1 conv** (kanal karışımı + kanal sayısı değişimi)
4. **BN + ReLU**

Yani “spatial mixing” shift ile, “channel mixing” 1×1 ile.

#### 3) `ShiftNetV2`

- Stem normal conv ile başlatıyor
- Sonra 3 stage var: her stage `ShiftConvBlock` + stride=2 ile çözünürlük düşürüp kanal artırıyor
- GAP + FC ile sınıflandırma

### Ne işe yarar?

- Mobil/edge tarzı senaryolarda:
  - 3×3 conv yükünü azaltmak
  - compute’u düşürmek ✅
- Spatial bilgi taşımayı “bedavaya yakın” yapıp öğrenmeyi 1×1’e bırakır.

### Dikkat

- Shift tek başına “öğrenmez”; öğrenme asıl olarak **1×1 conv**’da.
- Yön seçimi (`directions`) ve kanal bölme oranı performansı etkiler.
