# Convolutional Block Attention Module (CBAM)

## Amaç

CBAM, hem **kanal** hem de **uzamsal** dikkat mekanizmalarını birleştirerek feature map’leri iki boyutta da yeniden düzenler.

Yani şu iki soruyu cevaplar:

- Hangi kanal önemli?
- Görüntünün neresine bakılmalı?

---

## Temel Fikir

CBAM iki ardışık dikkat bloğundan oluşur:

1. Channel Attention
2. Spatial Attention

Bu iki mekanizma sırayla uygulanır.

---

## Nasıl Çalışır?

### 1) Channel Attention

- Global ortalama ve maksimum pooling
- Ortak MLP
- Kanal ağırlıkları üretimi

### 2) Spatial Attention

- Kanal bilgisi sıkıştırılır
- Uzamsal bir attention haritası üretilir

---

## Avantajları

✔ Kanal + uzamsal birlikte  
✔ Çeşitli görevlerde etkili  
✔ Görsel odaklanmayı artırır

---

## Dezavantajları

✘ SE/ECA’ya göre daha ağır  
✘ Hesaplama maliyeti artar

---

## Ne Zaman Kullanılır?

- Nesne tespiti
- Segmentasyon
- Görsel dikkat gerektiren görevler
