# CBAM Channel + Coordinate Attention (Enhanced Fusion)

## Amaç

Bu blok, klasik **CBAM Channel Attention** ile geliştirilmiş bir **Coordinate Attention+** yapısını birleştirir.  
Hedef:

> Kanal önemini dinamik öğrenirken aynı zamanda konum bilgisi taşıyan yönsel dikkat üretmek.

Bu yapı, yalnızca “hangi kanal önemli?” değil,  
**“bu kanal görüntünün neresinde önemli?”** sorusuna da cevap verir.

---

## Bu Yapı Neyi Geliştiriyor?

| Standart CBAM        | Bu Versiyon                        |
| -------------------- | ---------------------------------- |
| Kanal + spatial ayrı | **Channel + Coordinate hibrit**    |
| Sabit fusion         | **Öğrenilebilir fusion router**    |
| Konum bilgisi yok    | **X-Y yön profili kullanılır**     |
| Tek ölçekli spatial  | **Yerel + dilated yön filtreleri** |
| Sabit dikkat gücü    | **Sıcaklık + residual kontrollü**  |

---

## Mimari Bileşenler

### 1) ChannelAttentionFusionT

Bu modül kanal dikkatini klasik CBAM’den daha adaptif üretir.

✔ Avg ve Max pooling katkıları öğrenilir  
✔ Softmax router ile hangi istatistiğin daha önemli olduğu belirlenir  
✔ Attention keskinliği sıcaklık parametresi ile ayarlanır  
✔ Kanal etkileşimi girişe bağlı değişir

Bu yapı, kanal istatistiklerinin sabit toplanması yerine dinamik birleşmesini sağlar.

---

### 2) CoordinateAttPlus (Gelişmiş CA)

Bu modül, klasik Coordinate Attention’ın genişletilmiş bir versiyonudur.

✔ Yatay ve dikey yön profilleri ayrı çıkarılır  
✔ Yerel ve dilated yön filtreleri birlikte kullanılır  
✔ Kanal karıştırma (channel mixer) ile yönsel bilgi zenginleştirilir  
✔ H ve W için ayrı attention haritaları üretilir

Bu sayede konum bilgisi kaybolmaz, yönsel bağlam korunur.

---

### 3) Residual Kontrollü Attention

Dikkat doğrudan uygulanmaz. Bunun yerine:

> Orijinal bilgi ile attention çıktısı kontrollü şekilde karıştırılır.

Bu yaklaşım:

- Eğitimi stabilize eder
- Aşırı bastırmayı önler
- Attention yoğunluğunu ayarlanabilir kılar

---

### 4) Opsiyonel Spatial Gate

Ek bir uzamsal kapı mekanizması ile dikkat haritası daha da rafine edilebilir.  
Bu, dikkat uygulamasının ikinci bir kontrol katmanından geçmesini sağlar.

---

## Bu Yapı Ne Tür Attention’dır?

| Tür                      | Durum |
| ------------------------ | ----- |
| Channel Attention        | ✔     |
| Coordinate Attention     | ✔     |
| Directional Awareness    | ✔     |
| Multi-scale Spatial      | ✔     |
| Hybrid Attention         | ✔     |
| Residual Gated Attention | ✔     |

---

## Klasik CBAM’den Farkı

| Standart CBAM          | Bu Versiyon                    |
| ---------------------- | ------------------------------ |
| Avg+Max sabit birleşim | Router tabanlı fusion          |
| Spatial 7×7 conv       | Yönsel profil + dilated filtre |
| Konum bilgisi yok      | Koordinat tabanlı              |
| Direkt çarpım          | Residual kontrollü             |

---

## Ne Zaman Kullanılır?

- Nesne tespiti
- Küçük nesne yoğun sahneler
- Karmaşık arka planlı görüntüler
- Dikkat mimarisi araştırmaları

---

## Kısa Özet

Bu blok:

1. Kanal dikkatini dinamik öğrenir
2. Konum bilgisi taşıyan yönsel attention üretir
3. Multi-scale yönsel analiz yapar
4. Attention yoğunluğunu kontrol edilebilir kılar
5. Residual yapı ile stabil entegrasyon sağlar

> Klasik CBAM + Coordinate Attention’ın araştırma seviyesinde genişletilmiş hibrit versiyonudur.
