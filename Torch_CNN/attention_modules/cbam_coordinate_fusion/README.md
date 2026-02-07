# CBAM + Coordinate Attention Fusion

## Amaç

Bu klasör, **CBAM** ve **Coordinate Attention (CA)** bloklarının birlikte kullanıldığı
mimari varyasyonları içerir.

Amaç şudur:

> Kanal ve uzamsal dikkati birleştirirken aynı zamanda konumsal (directional) bilgiyi de koruyan bir attention yapısı tasarlamak.

CBAM genel görsel dikkati güçlendirirken, Coordinate Attention konum bilgisini kaybetmeden kanal etkileşimini modellemeye çalışır. Bu fusion yaklaşımı, her iki mekanizmanın güçlü yönlerini bir araya getirmeyi hedefler.

---

## Neden Fusion?

| Mekanizma            | Gücü                   | Zayıf Yönü                     |
| -------------------- | ---------------------- | ------------------------------ |
| CBAM                 | Kanal + uzamsal dikkat | Konum bilgisi sıkışabilir      |
| Coordinate Attention | Yön bilgisi korur      | Spatial harita üretimi sınırlı |

Fusion yaklaşımı ile:

- Kanal dikkat korunur
- Uzamsal odaklanma güçlenir
- Konum bilgisi kaybolmaz

---

## Genel Mimari Mantık

Fusion yapıları şu temel prensiplere dayanır:

1. Feature map alınır
2. Attention blokları farklı kombinasyonlarda uygulanır
3. Kanal ve uzamsal bilgi birlikte yeniden düzenlenir
4. Sonuç, daha bilgilendirici bir feature map olur

---

## Pattern Yapısı

Bu klasörde birden fazla mimari varyasyon bulunur:

| Pattern   | Açıklama                                          |
| --------- | ------------------------------------------------- |
| Pattern-1 | CBAM ve CA bloklarının temel birleşim denemeleri  |
| Pattern-2 | Daha gelişmiş fusion stratejileri ve varyasyonlar |

Her pattern klasörü içinde:

- `models/` → PyTorch implementasyonları
- `notes/` → Gelişim notebook’ları
- `assets/` → Diyagramlar ve görseller

---

## Bu Yapı Ne Tür Attention’dır?

| Tür                   | Durum |
| --------------------- | ----- |
| Channel Attention     | ✔     |
| Spatial Attention     | ✔     |
| Directional Awareness | ✔     |
| Hybrid Attention      | ✔     |

---

## Kullanım Senaryosu

Bu fusion yaklaşımları özellikle şu durumlarda anlamlı olabilir:

- Nesne tespiti (Detection)
- Küçük nesnelerin yoğun olduğu sahneler
- Konumsal hassasiyetin önemli olduğu görevler
- Zor sahnelerde feature seçiminin kritik olduğu durumlar

---

## Not

Bu klasör, dikkat mekanizmalarının birleşimlerini araştırma amacıyla içerir.
Her pattern, mimari keşif sürecinin farklı bir aşamasını temsil eder.
