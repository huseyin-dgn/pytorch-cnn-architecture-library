# CBAM Residual Dynamic Spatial Attention (Coord-Aware)

## Amaç

Bu blok, klasik **CBAM** mekanizmasını daha adaptif, konum duyarlı ve dinamik hale getirmek için tasarlanmıştır.  
Standart CBAM sabit kurallarla attention üretirken, bu yapı:

> **Girişe bağlı olarak değişen dikkat üretir.**

Yani attention artık statik değil, **veri koşullu (data-dependent)** bir yapıya sahiptir.

---

## Bu Yapı Neyi Geliştiriyor?

| Standart CBAM            | Bu Versiyon                                 |
| ------------------------ | ------------------------------------------- |
| Avg + Max sabit birleşim | **Öğrenilebilir fusion**                    |
| Tek uzamsal kernel       | **Çoklu kernel + dinamik seçim**            |
| Konum bilgisi yok        | **Koordinat bilgisi entegre**               |
| Sabit sigmoid            | **Sıcaklık kontrollü gating**               |
| Direkt çarpım            | **Residual kontrollü attention uygulaması** |

---

## Mimari Bileşenler

### 1) ChannelAttentionFusionT

Bu modül kanal dikkatini klasik CBAM’den daha esnek üretir.

✔ Avg ve Max pooling katkıları **öğrenilir**  
✔ Softmax ile iki kaynağın katkı oranı belirlenir  
✔ Attention keskinliği **sıcaklık (T)** ile kontrol edilir  
✔ Kanal etkileşimi dinamik hale gelir

Bu sayede model şunu öğrenir:

> “Bu giriş için hangi kanal istatistiği daha anlamlı?”

---

### 2) DynamicSpatialAttention (Coord-Aware)

Standart CBAM’de tek bir uzamsal filtre bulunur.  
Bu yapı ise:

✔ Birden fazla kernel boyutu  
✔ Opsiyonel dilated branch  
✔ Router ile **dinamik kernel seçimi**  
✔ X–Y koordinat bilgisi ile **konum duyarlı attention**

Bu mekanizma, farklı sahneler için farklı uzamsal dikkat stratejileri üretir.

---

### 3) Sıcaklık Kontrollü Dikkat

Hem kanal hem uzamsal dikkat için sıcaklık parametresi kullanılır.

| T değeri | Etki                   |
| -------- | ---------------------- |
| Küçük T  | Daha keskin attention  |
| Büyük T  | Daha yumuşak attention |

Bu, dikkat dağılımının ne kadar seçici olacağını kontrol eder.

---

### 4) Residual Attention Entegrasyonu

Attention çıktısı doğrudan uygulanmaz, residual karışım yapılır:

> Orijinal bilgi tamamen kaybolmaz, kontrollü şekilde dikkat eklenir.

Bu yaklaşım:

- Eğitimi daha stabil yapar
- Aşırı bastırmayı önler
- Attention yoğunluğunu ayarlamaya izin verir

---

## Bu Yapı Ne Tür Attention’dır?

| Tür                      | Durum |
| ------------------------ | ----- |
| Channel Attention        | ✔     |
| Spatial Attention        | ✔     |
| Multi-scale Spatial      | ✔     |
| Dynamic Attention        | ✔     |
| Hybrid Attention         | ✔     |
| Residual Gated Attention | ✔     |
| Coordinate-Aware         | ✔     |

---

## Ne Zaman Kullanılır?

- Nesne tespiti (Detection)
- Küçük nesnelerin yoğun olduğu sahneler
- Karmaşık arka planlar
- Attention mimarilerinin araştırıldığı çalışmalar

---

## Kısa Özet

Bu blok:

1. Kanal dikkatini öğrenilebilir şekilde üretir
2. Uzamsal dikkati girişe göre seçer
3. Konum bilgisini korur
4. Attention yoğunluğunu kontrol edilebilir yapar
5. Residual yapı ile stabil entegrasyon sağlar

> Klasik CBAM’in araştırma seviyesinde genişletilmiş, adaptif bir versiyonudur.
