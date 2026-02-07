# Squeeze-and-Excitation (SE)

## Amaç

SE bloğu, kanalların önemini öğrenerek feature map’leri kanal bazında yeniden ölçekler.

---

## Temel Fikir

Global bağlam kullanılarak kanal bağımlılığı modellenir.

---

## Nasıl Çalışır?

1. Global ortalama pooling
2. Küçültme + genişletme (MLP)
3. Kanal ağırlıkları üretimi
4. Feature map yeniden ölçeklenir

---

## Avantajları

✔ Basit  
✔ Düşük maliyet  
✔ Kolay entegre edilir

---

## Dezavantajları

✘ Uzamsal dikkat içermez
