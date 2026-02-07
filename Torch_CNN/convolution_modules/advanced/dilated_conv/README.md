## Dilated Convolution (Atrous Conv) — Kısa Açıklama

Dilated conv’un olayı: **kernel boyutu büyümeden receptive field büyütmek.**  
Yani 3×3 conv hâlâ 3×3 parametreyle çalışır ama örnekleme aralığı açılır.

### Dilation ne yapar?

- `dilation = 1` → normal 3×3
- `dilation = 2` → aralıklı 3×3 (sanki 5×5 gibi kapsar)
- `dilation = 4` → daha geniş alan (sanki 9×9 gibi kapsar)

Etkin kernel:
`k_eff = d*(k-1) + 1` (k=3 için → `k_eff = 2d + 1`)

### Bu kodda nerede kullanılıyor?

`DilatedBlock` içinde ikinci conv:

- `conv2`: **dilated 3×3**
  - `dilation = dilation param`
  - `padding = dilation` (stride=1 iken H/W korunması için)

Yani blok yapısı:

- `conv1`: normal conv (d=1)
- `conv2`: dilated conv (d=2 veya d=4)
- üstüne **residual (shortcut)** ekleniyor

### Network akışı (özet)

- `stem`: 3→16 normal conv
- `stage1`: d=1 (boyut sabit)
- `stage2`: stride=2 ile downsample + d=2
- `stage3`: stride=2 ile downsample + d=4
- sonra GAP + FC

### Ne işe yarar?

- Daha geniş bağlam (context) görür ✅
- Parametre sayısını patlatmadan “uzun menzilli” feature çıkarır ✅
- Segmentation / detection gibi işlerde sık kullanılır ✅

### Dikkat edilmesi gerekenler

- Büyük dilation bazen **gridding artifact** üretebilir (örnekleme seyrekleşir) ⚠️  
  Çözüm: farklı dilation’ları karıştırmak (ASPP gibi) veya stage’lere dengeli dağıtmak.
