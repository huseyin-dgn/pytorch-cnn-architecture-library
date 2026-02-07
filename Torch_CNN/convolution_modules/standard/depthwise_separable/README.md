## Depthwise Separable Convolution

Depthwise Separable Conv’un olayı: **klasik 3×3 conv’u iki parçaya bölüp ucuzlatmak.**  
Normal conv: hem uzamsal (spatial) hem kanal karışımını tek seferde yapar → pahalı.  
Burada ise:

1. **Depthwise (spatial)**: Her kanal kendi 3×3 filtresiyle işlenir
2. **Pointwise (channel mixing)**: 1×1 conv ile kanallar karıştırılır + kanal sayısı değiştirilir

### Bu kodda ne yapıyor?

#### `DepthwiseSeparableConv`

- `depthwise`: `groups = c_in`  
  → her input kanalına ayrı filtre (kanallar birbirine karışmaz)
- `pointwise`: 1×1 conv  
  → kanalları karıştırır ve `c_out`’a çıkarır

Sıra:
`x -> depthwise -> pointwise`

### Neden daha hafif?

Klasik conv maliyeti kabaca:

- `C_in * C_out * k*k`

Depthwise separable maliyeti:

- `C_in * k*k` (depthwise)
- `C_in * C_out` (pointwise)

Özellikle `k=3` ve `C_out` büyükken ciddi fark eder. 💸

### Model akışı (`TinyDepthwiseNet`)

- 3 blok arka arkaya: 3→32→64→128
- İki yerde MaxPool ile çözünürlük düşüyor (32→16→8)
- GAP + FC ile sınıflandırma

### Nerede kullanılır?

- MobilNet tarzı hafif mimariler 📱
- Edge cihazlar / düşük compute senaryoları ⚙️

### Dikkat

- Bu örnekte BN yok; pratikte genelde `Conv -> BN -> ReLU` şeklinde stabilize edilir.
- Çok hafiflediği için bazen kapasite düşer; blok sayısını artırarak dengelenir.
