## MBConv (Inverted Bottleneck)

MBConv’un olayı: **kanalı önce şişir (expand), uzamsal işlemi depthwise ile ucuza yap, sonra tekrar sıkıştır (project).**  
Bu yapı MobileNetV2/EfficientNet çizgisinin temel taşıdır. ⚙️📱

### Blok yapısı (sıra net)

1. **1×1 Expand**  
   `C_in -> C_exp = C_in * expansion_factor`  
   Amaç: kapasiteyi artırıp feature’ları zenginleştirmek.

2. **3×3 Depthwise (stride burada)**  
   `groups = C_exp` olduğu için her kanal ayrı filtre → ucuz spatial işlem.

3. **1×1 Project (Linear Bottleneck)**  
   `C_exp -> C_out`  
   Burada aktivasyon yok (linear), çünkü dar boğazda nonlinearity bilgi kaybını artırabiliyor.

4. **Residual (opsiyonel)**  
   Sadece `stride=1` ve `C_in == C_out` ise eklenir.  
   Downsample veya kanal değişimi varsa residual yok.

### Bu kodda neler var?

- Aktivasyon olarak **SiLU** kullanılmış (MobileNet/EfficientNet tarzı).
- Residual şartı doğru: `(stride == 1 and c_in == c_out)` ✅

### Model akışı (`MBConvNetSmall`)

- Stem: 3→16
- Stage1: 16→16 (2 MBConv, boyut sabit)
- Stage2: stride=2 ile 32→16 çözünürlük + 16→24 kanal
- Stage3: stride=2 ile 16→8 çözünürlük + 24→40 kanal
- GAP + FC

### Neden iyi?

- Normal conv’a göre **daha düşük FLOPs** hedefler (depthwise sayesinde)
- Expand sayesinde kapasiteyi korur/artar
- Residual ile optimizasyonu kolaylaştırır ✅

### Dikkat

- `expansion_factor` çok büyürse compute artar; çok küçülürse kapasite düşer.
- Pratikte EfficientNet’te buna ek olarak SE (Squeeze-Excite) ve drop connect de sık görülür.
