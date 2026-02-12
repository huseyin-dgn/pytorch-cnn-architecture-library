# DropBlock ve SpatialDropout

Bu dokümanda CNN mimarisi içinde kullanılan iki farklı regularization tekniği açıklanmaktadır:

- DropBlock
- SpatialDropout2d

Amaç, bu iki yöntemin nasıl çalıştığını ve aralarındaki temel farkları net biçimde ortaya koymaktır.

---

# 1) DropBlock

## Tanım

DropBlock, feature map üzerinde rastgele seçilen kare blokların tamamen sıfırlanması prensibiyle çalışan bir regularization tekniğidir.

Dropout’un uzamsal (spatial) olarak daha tutarlı versiyonu olarak düşünülebilir.

## Nasıl Çalışır?

- Feature map üzerinde bir merkez noktası seçilir.
- Belirlenen `block_size` kadar kare alan tamamen sıfırlanır.
- Bu işlem rastgele konumlarda uygulanır.
- Eğitim sürecinde drop oranı schedule ile artırılabilir.

## Temel Amaç

CNN'ler çoğu zaman lokal ve güçlü aktivasyonlara bağımlı hale gelir.  
DropBlock:

- Modelin tek bir bölgeye bağımlı olmasını engeller.
- Uzamsal korelasyonu kırar.
- Daha güçlü genelleme sağlar.

## Nerede Kullanılır?

- Orta ve derin katmanlarda
- Residual blok içinde
- Detection ve classification mimarilerinde

## Özellikler

- Blok bazlı düşürme
- Spatial olarak anlamlı
- Overfitting azaltmada etkilidir
- Küçük objelerde dikkatli kullanılmalıdır

---

# 2) SpatialDropout2d

## Tanım

SpatialDropout2d, feature map üzerinde rastgele kanalların (channel) tamamen sıfırlanması prensibiyle çalışan bir tekniktir.

Klasik Dropout’un CNN’e uygun versiyonudur.

## Nasıl Çalışır?

- Her batch için
- Belirli kanallar tamamen 0 yapılır
- O kanala ait tüm H×W aktivasyonları silinir

## Temel Amaç

Modelin tek bir feature kanalına bağımlı olmasını engellemek.

## Nerede Kullanılır?

- CNN blokları arasında
- Daha hafif regularization gerektiğinde
- Küçük ve orta ölçekli modellerde

## Özellikler

- Kanal bazlı düşürme
- Spatial blok silmez
- Hesap olarak hafiftir
- DropBlock kadar agresif değildir

---

# 3) Temel Farklar

| Özellik             | DropBlock       | SpatialDropout |
| ------------------- | --------------- | -------------- |
| Düşürme tipi        | Kare blok       | Kanal          |
| Uzamsal yapı        | Evet            | Hayır          |
| Agresiflik          | Daha yüksek     | Daha düşük     |
| Küçük objeye etkisi | Riskli olabilir | Daha güvenli   |
| Detection uyumu     | Daha uygun      | Orta           |

---

# 4) Mühendislik Perspektifi

DropBlock:

- Daha güçlü regularization
- Özellikle derin CNN’lerde etkili
- Feature map yapısını bilinçli bozar

SpatialDropout:

- Daha sade
- Daha stabil
- Küçük modeller için daha uygun

---

# 5) Ne Zaman Hangisi?

- Eğer model çok overfit ediyorsa → DropBlock
- Eğer hafif regularization istiyorsan → SpatialDropout
- Detection backbone’da → DropBlock daha yaygın
- Küçük dataset classification → SpatialDropout yeterli olabilir

---

# Sonuç

Her iki teknik de CNN içi regularization yöntemidir.

DropBlock daha yapısal ve güçlü bir müdahaledir.  
SpatialDropout ise daha hafif ve kanal seviyesinde çalışır.

Seçim, model mimarisi ve hedef göreve bağlıdır.
