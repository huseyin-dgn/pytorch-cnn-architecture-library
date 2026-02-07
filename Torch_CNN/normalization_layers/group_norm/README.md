# Group Normalization (GroupNorm) — Esnek Grup Seçimi (GroupNormFlex)

## Nedir?

**Group Normalization (GN)**, BatchNorm’dan farklı olarak batch boyutuna bağımlı değildir.  
Normalize işlemini **kanallar (C)** üzerinde gruplara bölerek yapar.

- **BN:** batch istatistikleri (N boyutu) ile çalışır → küçük batch’te sıkıntı
- **GN:** batch’e bakmaz, kanalı gruplara böler → küçük batch’te daha stabil

Bu klasördeki implementasyon GN’i “tek tip” değil, **LN/IN benzeri modlara da kayabilecek şekilde** esnek tasarlıyor.

---

## Bu implementasyonda ana fikir

### 1) `GNConfig` ile davranışı yönet

GN’in üç modu var:

- `mode="gn"` → klasik GroupNorm
- `mode="ln_like"` → **LayerNorm benzeri** (groups=1)
- `mode="in_like"` → **InstanceNorm benzeri** (groups=C)

Bu, tek bir sınıfla “GN / LN / IN” davranışlarını taklit edebilmeni sağlar.

---

## “Auto groups” ne yapıyor?

`groups="auto"` seçilirse, `choose_gn_groups()` kanala (C) en uygun grup sayısını seçer.

Öncelik listesi:

`(32, 16, 8, 4, 2, 1)`

Mantık:

- mümkünse 32 grup (klasik pratik)
- C 32’ye bölünmüyorsa 16 dene
- sonra 8,4,2,1

Amaç:

> “Group sayısı çok büyük olmasın ama kanal da boşa bölünmesin.”

**Kural:** `C % groups == 0` olmak zorunda.

---

## GroupNormFlex ne sağlıyor?

`GroupNormFlex(num_channels, config)`:

- config’e göre group sayısını belirler
- `nn.GroupNorm(groups, C)` çağırır
- forward’da direkt GN uygular

Avantaj:

- modelin farklı yerlerinde farklı C değerleri olsa bile “group seçimi” otomatik akar
- LN/IN benzeri davranışlar tek sınıftan kontrol edilir

---

## ConvGNAct bloğu (pratik kullanım)

`ConvGNAct` = **Conv → GN → Activation**.

Burada amaç:

- BN yerine GN kullanarak backbone benzeri blok üretmek
- activation seçimini (relu/silu/gelu/none) esnek tutmak

Bu blok, repo içinde başka modüllere de “drop-in” şekilde takılabilir.

---

## GNCNN: GN ile basit backbone örneği

Bu klasördeki örnek ağ:

- stem → stage1 → stage2
- her yerde ConvGNAct
- global average pooling + linear classifier

Ayrıca `return_features=True` ile P3/P4/P5 gibi ara feature map’leri de döndürebiliyor.

Bu, GN’in:

- sadece tek layer değil,
- bir backbone içinde nasıl davrandığını göstermek için.

---

## Ne zaman GN kullanılır?

✅ Batch küçükse (1–8 arası)  
✅ multi-GPU SyncBN yoksa  
✅ BN kararsızsa / train-eval farkı can sıkıyorsa  
✅ detection/segmentation gibi küçük batch ile eğitilen işlerde

❌ Batch büyük ve BN zaten stabilse  
❌ hızlı inference için BN fusion gibi optimizasyonlar hedefleniyorsa (GN daha “sabit maliyetli”)

---

## Pratik öneriler

- Klasik GN: `groups=32` (uyarsa)
- Kanal küçükse:
  - C=16 → groups=8/4/2
  - C=8 → groups=4/2/1
- Bu implementasyondaki auto seçimi bu pratikleri zaten otomatik yapıyor.

---

## Kısa Özet

Bu modül:

- GroupNorm’u batch bağımsız stabil bir normalizasyon olarak sunar
- `GNConfig` ile GN/LN/IN benzeri modlar sağlar
- `auto groups` ile C’ye göre uygun grup seçer
- ConvGNAct + GNCNN ile gerçek kullanım örneği verir
