## CoordConv (Coordinate Convolution)

CoordConv’un fikri çok net: **Conv katmanına “ben görüntünün neresindeyim?” bilgisini veriyorsun.**  
Normal conv translation-invariant çalışır; yani “bu feature sol üstte mi sağ altta mı?” bilgisini doğrudan taşımaz. CoordConv bunu çözer: input’a **(x, y)** (opsiyonel **r**) koordinat kanalları ekler. 🧭

### Bu kodda CoordConv nasıl yapılmış?

1. `cords(...)` fonksiyonu:

- `(H, W)` için `[-1, 1]` aralığında grid üretir
- `xx` ve `yy` kanalları çıkar:
  - `xx`: yatay konum (sol=-1, sağ=+1)
  - `yy`: dikey konum (üst=-1, alt=+1)
- `add_rad=True` ise bir de:
  - `rr = sqrt(x^2 + y^2)` (merkeze uzaklık) ekler

2. `CordConv.forward(x)`:

- `coords = cords(...)` üretir
- `x = cat([x, coords], dim=1)` ile input kanalına ekler
- Sonra standart `Conv2d` uygular

Yani Conv artık şunu görüyor:

- görüntü özellikleri + **mutlak konum bilgisi** ✅

### Neden işe yarar?

Özellikle şu görevlerde fark yaratır:

- (x,y) regresyon / keypoint / obje konumu tahmini 🎯
- “Nesne hep üst bölgede olur” gibi konuma bağlı pattern’ler
- Spatial reasoning işleri

Senin model tam bu sınıfa giriyor:

- Çıkış: `fc -> tanh -> (B,2)` yani **[-1,1] aralığında x,y tahmini**

### Bu mimaride CoordConv nereye konmuş?

- Backbone normal conv ile feature çıkarıyor
- **Head kısmında CoordConv var**
  - yani “yüksek seviye feature” üstüne konum bilgisi enjekte ediliyor ✅
- Head’in sonunda ayrıca **SE (Squeeze-Excite)** var:
  - kanalları ağırlıklandırıp önemli kanalları öne çekiyor

### Dikkat

- CoordConv konum bilgisini “kolaylaştırır”; bazı task’larda performansı ciddi artırır,
  ama her problemde şart değil.
- `add_rad=True` genelde merkez/uzaklık ilişkisi olan işlerde ekstra fayda sağlar.
