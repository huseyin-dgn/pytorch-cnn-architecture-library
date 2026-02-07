## Ghost Convolution — Kısa Açıklama

GhostConv’un amacı basit: **aynı kanal sayısını daha ucuza üretmek.**  
Normal conv ile bütün feature’ları “pahalı” şekilde üretmek yerine:

- az bir kısmını gerçek conv ile üret
- kalanını **ucuz işlemlerle (depthwise)** türet
- sonra birleştir ✅

### Bu kodda ne oluyor?

**GhostConv = Primary + Cheap + Concat**

- **Primary conv (pahalı kısım)**  
  `c_in -> c_int` üretir.  
  `c_int = round(c_out / ratio)`  
  Yani `ratio` büyüdükçe gerçek conv’un ürettiği kanal azalır → daha hafif 💸

- **Cheap conv (ucuz kısım, depthwise)**  
  Primary çıktısından `c_ghost = c_out - c_int` kadar “ghost feature” üretir.  
  Burada `groups=c_int` olduğu için depthwise çalışır → maliyet düşük ⚙️

- **Concat**  
  `[primary, ghost]` birleştirilir → hedef `c_out` tamamlanır.

### GhostBlock ne yapıyor?

- Ana yol: `GhostConv(3x3)` (stride ile downsample da yapabiliyor)
- Üstüne **residual shortcut** ekliyor:
  - `stride=1` ve `c_in=c_out` ise identity
  - değilse 1×1 conv ile boyut/kanal eşitleniyor
- Sonuna bir `ReLU` daha basıyor (istersen bu ekstra ReLU’yu kaldırabilirsin)

### Network akışı (özet)

- Stem: normal conv ile 3→16
- Stage1: GhostBlock 16→16 (2 blok)
- Stage2: downsample + 16→32
- Stage3: downsample + 32→64
- GAP + FC

### Ne işe yarar?

- Mobil/edge için: **daha az parametre + daha az FLOPs** hedefi 🎯
- Benzer representasyonu daha hafif şekilde çıkarmaya çalışır.
- Uygun yerde doğru ayarla iyi iş çıkarır; körlemesine her yere basmak şart değil.

### Dikkat

- `ratio` çok büyürse model “ucuz” feature’a fazla yaslanır → kalite düşebilir. ⚠️  
  Genelde 2 iyi başlangıçtır.
