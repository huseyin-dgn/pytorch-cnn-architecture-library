import torch
import torch.nn as nn

class DropPath(nn.Module):
    def __init__(self,
                 max_drop:float=0.0,
                 layer_idx:int | None = None,
                 num_layers:int | None = None,
                 warmup_steps:int = 0,
                 batchwise:bool = True,
                store_mask:bool = True ):
        super().__init__()
        self.max_drop = max_drop
        # max_drop: float = 0.0 :: DropPath’in ulaşabileceği maksimum drop oranı..
        # 0.1 demek: her forward’da residual yolun %10 ihtimalle kapatılması.

        self.layer_idx = layer_idx
        # Bu DropPath’in bağlı olduğu bloğun derinlik indeksi.
        # Eğer None ise: katmana göre ölçekleme yapılmaz, direkt max_drop kullanılır.

        self.num_layers = num_layers
        # Anlamı: Schedule uygulayabilmek için toplam kaç blok/layer olduğunu söyler.
        # layer_idx ile beraber çalışır.
        # num_layers <= 1 veya None ise schedule devre dışı kalır

        self.warmup_steps = int(warmup_steps)
        # DropPath’in gücünü ilk N adımda yavaş yavaş artırmak.
        # Training başında modeli destabilize etmemek için kullanılır.

        self.batchwise = bool(batchwise)
        # Maskeyi nasıl uygulayacağın.
            ## True ise :: her örnek için: residual yol ya tamamen açık ya tamamen kapalı
            ## False ise :: element-wise dropout’a benzer (her piksel/eleman ayrı düşebilir)
                ## bu artık “DropPath” değil, daha çok Dropout benzeri davranır.

        self.store_mask = bool(store_mask)
        # Debug için son kullanılan maskeyi self.last_mask içine saklasın mı?
            ## True yaparsan:
                # self.last_mask dolu olur (detach edilip saklanır)
                # görselleştirme / debug kolaylaşır
            ## Normal eğitimde genelde False

        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.last_mask = None
    
    def layer_scaled_drop(self) -> float:
            # Eğer derinlik bazlı schedule bilgisi yoksa:
                # - layer_idx None ise (hangi katmanda olduğumuzu bilmiyorsak)
                # - num_layers None ise (toplam kaç katman var bilmiyorsak)
                # - num_layers <= 1 ise (tek katmanlı bir yapıysa)
            # Bu durumda derinliğe göre ölçekleme YAPILMAZ,
            # DropPath oranı sabit olarak max_drop kullanılır.
            if self.layer_idx is None or self.num_layers is None or self.num_layers <= 1:
                return self.max_drop

            # layer_idx: şu an hangi residual bloktayız (0-indexed)
            # num_layers - 1:
            #   çünkü layer index'leri 0'dan başlar
            #   en derin blok için:
                #       layer_idx = num_layers - 1
                #       frac = 1.0  (maksimum drop'a ulaşsın diye)
            # frac: modelin DERİNLİĞİNDE ne kadar ilerlediğimizi gösteren oran
            frac = float(self.layer_idx) / float(self.num_layers - 1)
            # Gerçek uygulanacak DropPath oranı:
            # - sığ katmanlarda frac küçük → drop küçük
            # - derin katmanlarda frac büyük → drop büyük
            # Örnek:
                #   num_layers = 6
                #   layer_idx = 2  (0-index)
                #   frac = 2 / 5 = 0.4
                #   max_drop = 0.1
                #   drop = 0.1 * 0.4 = 0.04
            return self.max_drop * frac
        
    def warmup_drop_prob(self, drop: float) -> float:
            # drop: burada warmup uygulanmadan önceki HEDEF DropPath oranı.
            # Yani drop dediğimiz şey direkt bir oran/olasılık:
            #   - "bu yolu kaç ihtimalle kapatayım?" (ör: 0.1 = %10)
            
            # warmup_steps: bu hedef orana KAÇ ADIMDA ulaşacağımız.
            #   - warmup_steps = 1000 ise, DropPath gücü ilk 1000 step'te 0'dan hedefe çıkar.
            # Mantık: eğitim başında DropPath'i bir anda açmak yerine yavaş yavaş artırıyoruz.

            # Eğer warmup_steps 0 veya daha küçükse warmup yok:
            # direkt hedef oran olan drop'u geri döndür.
            if self.warmup_steps <= 0:
                return drop

            # s: o anki step (şu ana kadar kaç tane forward / eğitim adımı geçti?)
            s = int(self.step.item())

            # t: warmup ilerleme oranı.
                #   - t = s / warmup_steps
                #   - clamp (0..1) yapıyoruz:
                #       max(..., 0.0)  -> negatif olmasın
                #       min(..., 1.0)  -> 1'i geçmesin
            # Örnek:
                #   s=200, warmup_steps=1000  -> t=0.2 (warmup'ın %20'si)
            t = min(max(s / float(self.warmup_steps), 0.0), 1.0)

            # effective_drop = drop * t
            # Yani o anki step'te uygulanacak DropPath olasılığı.
            # Örnek:
                #   drop=0.1, t=0.2  -> 0.02
                #   => bu step'te yolu %2 ihtimalle kapatacağız.
            return drop * t

    def current_drop_prob(self) -> float:
        # current_drop_prob() dediğimiz şey aslında 
                # “şu anda uygulanacak gerçek (efektif) drop oranı kaç?” sorusunun cevabı.
        d = self.layer_scaled_drop() # Katmana göre hedef drop’u buluyor
        d = self.warmup_drop_prob(d) # Warmup uygular (step’e göre yavaş yavaş açar)

        # Güvenlik clamp’i yapar
        if d < 0.0:
            d=0.0
        if d >= 1.0:
            d = 0.999
        return d
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eğer train moddaysak step sayacını artırıyoruz.
        # step dediğimiz şey: o ana kadar kaç tane training forward (batch) geçtiği.
        if self.training:
            self.step += 1

        # current_drop_prob() bize "şu an bu adımda uygulanacak gerçek drop olasılığı"nı döndürür.
        # Bu değer:
        #   - layer schedule (derinliğe göre) + warmup (step'e göre) ile belirlenir.
        drop_prob = self.current_drop_prob()

        # Eğer drop_prob 0 veya daha küçükse ya da eval moddaysak:
        # DropPath kapalıdır, inputu olduğu gibi döndürürüz.
        if drop_prob <= 0.0 or (not self.training):
            if self.store_mask:
                self.last_mask = None
            return x

        # keep_prob: yolun AÇIK kalma olasılığı
        # drop_prob: yolun KAPALI olma olasılığı
        keep_prob = 1.0 - drop_prob

        # batchwise=True ise:
        # - her örnek için tek bir 0/1 karar verilir (path tamamen açık / tamamen kapalı)
        # - bu yüzden mask şekli (B,1,1,1...) olur
        #
        # batchwise=False ise:
        # - element-wise maske olur (dropout gibi), x'in tüm elemanlarına ayrı ayrı uygulanır
        if self.batchwise:
            mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B,1,1,1...)
        else:
            mask_shape = x.shape  # element-wise

        # Bernoulli(keep_prob) maske üret:
        # - keep_prob olasılıkla 1 (açık)
        # - (1-keep_prob)=drop_prob olasılıkla 0 (kapalı)
        mask = torch.empty(mask_shape, device=x.device, dtype=x.dtype).bernoulli_(keep_prob)

        # Debug amaçlı maskeyi saklamak istersek (detach ederek)
        if self.store_mask:
            self.last_mask = mask.detach()

        # Expected value düzeltmesi:
        # - mask=0 ise çıktı 0 olur (residual yol kapandı)
        # - mask=1 ise çıktı x/keep_prob olur (ölçekleme ile beklenen değer korunur)
        return x * mask / keep_prob



                