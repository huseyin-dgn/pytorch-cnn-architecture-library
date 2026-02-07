# SyncBatchNorm (SyncBN)

## Nedir?

**SyncBatchNorm**, klasik `BatchNorm2d`’nin çoklu GPU (Distributed Data Parallel / DDP) eğitimde yaşadığı temel sorunu çözer:

> Her GPU kendi mini-batch’inin istatistiğini hesapladığı için, “global batch” küçükmüş gibi davranır.

Örnek:

- 4 GPU var, her GPU batch=4 görüyor → her GPU BN istatistiği batch=4 üzerinden çıkar
- toplam efektif batch 16 olsa bile BN **4 gibi** davranır

**SyncBN** bu istatistikleri GPU’lar arasında senkronlayarak:

- mean/var’ı **tüm GPU’ların batch’leri üzerinden** hesaplar
- böylece BN davranışı global batch’e daha yakın olur

---

## Bu dosya neyi gösteriyor?

Bu klasördeki model bir “BN tabanlı backbone” örneği.

### Yapı

- `ConvBNAct`: Conv → BN → SiLU
- `ResidualBlock`: BN’li residual blok
- `SyncBNNet`: stem + 3 stage + GAP + FC

Şu an kod içinde BN var:

- `nn.BatchNorm2d(...)`

ve yorum olarak: “sonra SyncBN’e çevireceğiz”.

Yani dosyanın amacı:

- model mimarisini göstermek
- sonra training tarafında BN yerine SyncBN kullanma fikrini taşımak

---

## SyncBN nasıl uygulanır? (konsept)

Bu repo düzeyinde mantık şu:

1. Modeli normal BN ile yaz (bu dosyada olduğu gibi)
2. DDP ile çoklu GPU eğitimi yapacaksan:
   - BN katmanlarını **SyncBatchNorm**’a çevir
3. Eğitim sırasında istatistikler tüm GPU’larda ortak hesaplanır

> Tek GPU veya CPU’da SyncBN’in ekstra faydası yoktur.

---

## Ne zaman SyncBN gerekli?

✅ Multi-GPU DDP eğitimi yapıyorsan  
✅ GPU başına batch küçükse (özellikle 1–8)  
✅ BN kullanmak istiyorsun ama istatistik gürültüsü yaşıyorsun

❌ Tek GPU eğitimde (boşuna overhead)  
❌ Zaten GN/LN/IN gibi batch bağımsız norm kullanıyorsan  
❌ Çok küçük batch + çok değişken dağılım varsa, GN bazen daha iyi “direct çözüm” olur

---

## SyncBN vs Alternatifler

- **SyncBN:** BN’in mantığını korur, çoklu GPU’da istatistik sorununu düzeltir
- **GN/LN:** batch bağımsız → Sync gerektirmez, ama BN ile aynı davranışı vermez
- **FrozenBN:** istatistiği dondurur → fine-tune’da iyi, sıfırdan training’de şart değil

---

## Kısa Özet

Bu modül:

- BN’li bir CNN backbone örneği içerir
- Multi-GPU eğitimde BN’in lokal batch problemine çözüm olarak **SyncBatchNorm** fikrini hedefler
- DDP altında SyncBN ile BN’i “global batch” gibi çalıştırarak stabiliteyi artırır
