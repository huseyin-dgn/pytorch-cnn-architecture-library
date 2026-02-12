import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock2D_Scheduled(nn.Module):
    def __init__(self, max_drop_prob=0.1, block_size=7, total_steps=10_000):
        # total_steps :: :: Blok düşürme işlemi ne kadar sürede tam güce çıksın.
        # total_steps = 10.000 :: :: DropBlock ilk 10.000 öğrenme adımında yavaş yavaş güçlenecek.Sonrasında sabit kalacak.

        '''
        | Adım        | DropBlock gücü    |
        | ----------- | ----------------- |
        | Step 0      | Hiç karalama yok  |
        | Step 2.500  | Az karalama       |
        | Step 5.000  | Orta karalama     |
        | Step 10.000 | Maksimum karalama |
        | Step 20.000 | Hâlâ maksimum     |
        total_steps aslında “eğitimin yüzde kaçı süresince DropBlock artsın?” demek
        '''

        super().__init__()
        self.max_drop_prob = float(max_drop_prob)
        self.block_size = int(block_size)
        self.total_steps = int(total_steps)
        self.register_buffer("step", torch.zeros((), dtype=torch.long)) 
        # step :: ::  Model şu ana kadar kaç kere öğrenme güncellemesi yaptığı.

    def _current_drop_prob(self):
        # step buffer (tensor) -> Python int: kaçıncı training iterasyonundayız?
        s = int(self.step.item())

        # Schedule ilerleme oranı (0..1):
        # t = s/total_steps  -> 0: başlangıç, 1: schedule bitti
        # max(1, total_steps): total_steps=0 yazılırsa bölme patlamasın
        # clamp: t'yi 0 ile 1 arasında tut
        t = min(max(s / max(1, self.total_steps), 0.0), 1.0)

        # DropBlock şiddeti: 0 → max_drop_prob lineer artar
        return self.max_drop_prob * t

    def forward(self, x):
        if self.training:
            self.step += 1 # Eğer forward akışı eğitim aşamasında ise step her akışta 1 artsın.

        drop_prob = self._current_drop_prob() # Yukarıdaki fonksiyondan belirlenen drop_prob değerini alır.

        if (not self.training) or drop_prob <= 0.0:
            return x # Eğitim aşamasında değilse veya drop_prob sıfır veya negatifse, girdi tensorünü olduğu gibi döndürür. Yani DropBlock uygulanmaz.

        if x.dim() != 4:
            raise ValueError(f"DropBlock 4D tensor bekler (N,C,H,W). Geldi: {x.shape}")
            # DropBlock genellikle görüntü verileri üzerinde kullanılır ve bu veriler genellikle 4 boyutludur: (batch_size, channels, height, width). Eğer girdi tensorü bu formatta değilse, bir hata mesajı verir.

        n, c, h, w = x.shape # Girdi tensorünün boyutlarını n (batch size), c (kanal sayısı), h (yükseklik) ve w (genişlik) olarak ayırır.

        bs = min(self.block_size, h, w)
        if bs < 1:
            return x # Eğer block_size, girdi tensorünün yüksekliği veya genişliğinden büyükse, block_size'ı bu boyutlara eşitler. Eğer block_size 1'den küçüksa, girdi tensorünü olduğu gibi döndürür.

        # valid_h ve valid_w, 7×7 bloğu feature map’in içinde taşmadan kaç farklı konuma yerleştirebileceğimi söyler. +1 ise ilk konumu bize söyler.
        valid_h = h - bs + 1 # DropBlock'un geçerli olduğu yükseklik ve genişlik hesaplanır. Bu, block_size'ın etkisiyle düşürülen boyutları temsil eder.
        valid_w = w - bs + 1 # DropBlock'un geçerli olduğu genişlik hesaplanır.
        block_area = bs * bs # DropBlock'un kapladığı alan hesaplanır.

        gamma = drop_prob * (h * w) / (block_area * valid_h * valid_w) 
        # DropBlock'un uygulanacağı merkez noktalarının olasılığı hesaplanır. 
        # Bu, drop_prob'un block_area ve geçerli konum sayısına göre ayarlanmasıyla elde edilir.
        # Geriye kalan alanların ve uygulanan alanların bölümünü alarak yerleşme oranlarını belirledik

        '''
        Mesela köşeden 3x3 lük kalan bir alan var.Biz gittik orayı merkez konum aldık.Olmaz çünkü 7x7 burada iş yapamaz.Sadece 7×7 bloğun TAMAMEN sığabileceği başlangıç/merkez konumları dikkate alınır.
        Asıl amaç 
            - Alan bazlı bir hedefi, konum bazlı bir olasılığa çevirmek.
            Yani:
            Bizim hedefimiz:
                - “feature map’in yaklaşık %X’i düşsün” (alan bazlı)
            Ama DropBlock’un çalıştığı mekanizma:
                - “hangi konumlar blok merkezi olsun?” (konum bazlı)
            Gamma bu iki dünyayı birbirine bağlayan şey.
        '''

        center_mask = (torch.rand((n, c, valid_h, valid_w), device=x.device) < gamma).float()
        # Her (batch, kanal, konum) için 0 ile 1 arasında rastgele bir sayı üretir.
        # torch.rand((n, c, valid_h, valid_w)) :: :: İçerik: her hücre için 0 ile 1 arasında rastgele say = ör: 0.13, 0.87, 0.004, 0.56 ...
        # rand < gamma :: :: “Bu konum, gamma olasılığıyla seçilsin mi? Somut düşünce:
            # gamma = 0.001 olsun.
                # rand sayısı:
                # 0.0004 → seçildi
                # 0.2 → seçilmedi
                # 0.0009 → seçildi
                # 0.8 → seçilmedi
        # center_mask, her (n,c,h,w) merkez adayı için 0–1 arası random sayı üretip, rand < gamma olan yerleri 1 yaparak “blok merkezleri burada” diye işaretliyor.

        # Bu padding feature map e değil , center_mask a uygulanır. Çünkü center_mask, block_size'ın etkisiyle düşürülen boyutlara göre oluşturulmuştur.
        #  Padding, center_mask'ı orijinal girdi boyutlarına geri getirmek için yapılır. Bu, blokların tam olarak uygulanabilmesi için gereklidir.
        pad_h = (bs - 1) // 2 # center_mask'ı, block_size'ın etkisiyle düşürülen boyutlara göre doldurur. Bu, blokların tam olarak uygulanabilmesi için gereklidir.
        pad_w = (bs - 1) // 2 # center_mask'ı, block_size'ın etkisiyle düşürülen boyutlara göre doldurur. Bu, blokların tam olarak uygulanabilmesi için gereklidir.

        center_mask = F.pad(center_mask, (pad_w, pad_w, pad_h, pad_h))
        # PyTorch F.pad (4D tensor için) şu sırayı kullanır: (W sol, W sağ, H üst, H alt)
            ## W’ye: pad_w sol + pad_w sağ
            ## H’ye: pad_h üst + pad_h alt
        '''
                pad=3 ise
        center_mask boyutu (n, c, valid_h, valid_w) idi
                pad sonrası:
        (n, c, valid_h + 2*3, valid_w + 2*3) = (n, c, valid_h+6, valid_w+6)
        “maskenin etrafına boş çerçeve koymak” :: :: böylece 7×7 yayılma kenarda bile tam yapılabiliyor. 
        '''

        '''
            Ne var elimizde?
            center_mask (5x5)
                0 0 0 0 0
                0 0 1 0 0
                0 0 0 0 0
                0 0 0 0 0
                0 0 0 0 0

            Ne yapmak istiyoruz?
                - Bu tek noktayı, etrafına 7×7 (örnekte küçük diye 3×3 düşün) blok haline getirmek.
        '''

        block_mask = F.max_pool2d(center_mask, kernel_size=bs, stride=1, padding=bs // 2)
        # center_mask :: :: 1 = “burada merkez var -- 0 = “merkez yok” :: :: max_pool2d burada pooling için değil, yayma (dilation) için kullanılıyor.
        # center_maskte 1 olan merkez noktalarını, etrafına 7×7 kare alan olarak yaymak.
        # padding = bs // 2 :: :: Böylece bloklar tam olarak uygulanabilir. Örneğin, block_size=7 ise, padding=3 olur ve bu da blokların kenarlara tam olarak yayılmasını sağlar.
        
        mask = 1.0 - block_mask # element-wise : Matristen tek seferde çıkarmıyoruz; her elemandan 1 çıkarıyoruz.
        # block_mask'ta 1 olan yerler blok merkezi olduğu için düşürülecek bölgeyi temsil eder. Bu yüzden 1.0 - block_mask yaparak, tutulacak yerleri 1.0 ve düşürülecek yerleri 0.0 yapıyoruz.
        # block_mask: (B, C, H, W) boyutunda float tensor
        # - Değerleri genelde 0.0 ve 1.0 olur
        # - 1.0 => "drop edilecek bölge" (blok içinde)
        # - 0.0 => "drop yok"

        # Ama biz x ile çarparken şu mantığı isteriz:
        # - tutulacak yerler 1.0
        # - düşürülecek yerler 0.0
        # Bu yüzden ters çeviriyoruz (invert):

        keep_prob = mask.mean().clamp(min=1e-6)
        # mask.mean() :: :: maskede 1 olanların oranı = “ne kadarını tuttuk?”
        # Maskede kaç hücrenin “kaldığını” (1) ortalama olarak ölçüp, bölme yaparken sıfıra yaklaşmasın diye taban koymak.
        return x * mask / keep_prob
    
    # Feature map: 7×17 ise 

        # █████████████████
        # █████████████████
        # █████████████████
        # █████████████████
        # █████████████████
        # █████████████████
        # █████████████████

    # DropBlock 7×7 blok ise

        # ███████..........
        # ███████..........
        # ███████..........
        # ███████..........
        # ███████..........
        # ███████..........
        # ███████..........