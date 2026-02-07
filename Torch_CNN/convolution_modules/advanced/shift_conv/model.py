import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# 1) SHIFT OPERATÖRÜ (Parametresiz Spatial Mixing)
# ------------------------------------------------------------
def shift2d(
    x: torch.Tensor,
    directions=None,
    mode: str = "zero_pad"
) -> torch.Tensor:
    """
    ShiftConv'un kalbi: Kanalların bir kısmını sağa/sola/yukarı/aşağı kaydırır.
    Bu sayede 3x3 conv gibi komşuluk bilgisi taşınır ama PARAMETRE YOKTUR.

    x: (B, C, H, W)

    directions: [(dx, dy), ...]
      dx: +1 => sağa kaydır, -1 => sola
      dy: +1 => aşağı kaydır, -1 => yukarı
      (0,0) => hiç kaydırma

    mode:
      - "zero_pad": torch.roll'un wrap-around etkisini sıfırlayıp "padding" gibi davranır.
      - "circular": wrap-around kalsın (genelde istenmez, ama debug için yararlı olabilir)

    Mantık:
      1) Kanalları G= len(directions) parçaya böleriz.
      2) Her parçayı farklı yöne kaydırırız.
      3) Tekrar kanalda birleştiririz.
    """
    if directions is None:
        # 5 yön: merkez + 4 komşu
        directions = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

    B, C, H, W = x.shape
    G = len(directions)

    # Kanalları G parçaya böleceğiz.
    # Kanal sayısı G'ye bölünmüyorsa son parça biraz daha büyük olur (problem değil).
    base = C // G
    sizes = [base] * (G - 1) + [C - base * (G - 1)]
    chunks = torch.split(x, sizes, dim=1)

    shifted_chunks = []
    for chunk, (dx, dy) in zip(chunks, directions):
        if dx == 0 and dy == 0:
            shifted_chunks.append(chunk)
            continue

        # torch.roll wrap-around yapar: taşan kısım diğer taraftan geri girer.
        y = torch.roll(chunk, shifts=(dy, dx), dims=(-2, -1))

        if mode == "zero_pad":
            # Wrap-around bölgelerini sıfırla => "zero padding" gibi davranır.

            # Dikey sıfırlama
            if dy > 0:   # aşağı kaydı => üstte boşluk oluştu
                y[..., :dy, :] = 0
            elif dy < 0: # yukarı kaydı => altta boşluk oluştu
                y[..., dy:, :] = 0

            # Yatay sıfırlama
            if dx > 0:   # sağa kaydı => solda boşluk oluştu
                y[..., :, :dx] = 0
            elif dx < 0: # sola kaydı => sağda boşluk oluştu
                y[..., :, dx:] = 0

        shifted_chunks.append(y)

    return torch.cat(shifted_chunks, dim=1)


# ------------------------------------------------------------
# 2) SHIFT + 1x1 CONV BLOĞU
# ------------------------------------------------------------
class ShiftConvBlock(nn.Module):
    """
    ShiftConvBlock = Shift (parametresiz spatial mixing) + 1x1 Conv (öğrenilebilir channel mixing)

    Neden 1x1 conv şart gibi?
      Shift kanalı "konum" olarak karıştırır ama kanallar arası öğrenilebilir karışım sağlamaz.
      1x1 conv => kanallar arası öğrenilebilir karışım (mixing) sağlar.
    """
    def __init__(self, cin, cout, directions=None, stride=1, shift_mode="zero_pad"):
        super().__init__()
        self.directions = directions
        self.shift_mode = shift_mode

        # Downsample: Shift sonrası çözünürlüğü düşürmek istiyorsak basit bir avgpool kullanıyoruz.
        # (İstersen bunun yerine stride'lı 1x1 conv da yapılabilir.)
        self.down = nn.Identity() if stride == 1 else nn.AvgPool2d(kernel_size=stride, stride=stride)

        # Pointwise (1x1) conv: kanal karışımı + kanal sayısını değiştirme
        self.pw = nn.Conv2d(cin, cout, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = shift2d(x, directions=self.directions, mode=self.shift_mode)
        x = self.down(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# ------------------------------------------------------------
# 3) NET BİR MODEL: ShiftNetV2
# ------------------------------------------------------------
class ShiftNetV2(nn.Module):
    """
    Basit ama anlaşılır Shift tabanlı CNN:
      Stem: klasik 3x3 (girişten ilk feature)
      Stage1-3: ShiftConvBlock (stride=2 ile downsample)
      Head: GAP + FC
    """
    def __init__(self, in_channels=3, num_classes=10, directions=None):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.stage1 = ShiftConvBlock(32,  64,  directions=directions, stride=2)
        self.stage2 = ShiftConvBlock(64,  128, directions=directions, stride=2)
        self.stage3 = ShiftConvBlock(128, 256, directions=directions, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, verbose=False):
        if verbose: print("input :", x.shape)

        x = self.stem(x)
        if verbose: print("stem  :", x.shape)

        x = self.stage1(x)
        if verbose: print("s1    :", x.shape)

        x = self.stage2(x)
        if verbose: print("s2    :", x.shape)

        x = self.stage3(x)
        if verbose: print("s3    :", x.shape)

        x = self.pool(x)
        if verbose: print("gap   :", x.shape)

        x = torch.flatten(x, 1)
        if verbose: print("flat  :", x.shape)

        x = self.fc(x)
        if verbose: print("logits:", x.shape)

        return x


# ------------------------------------------------------------
# 4) ÇALIŞTIRMA
# ------------------------------------------------------------
if __name__ == "__main__":
    model = ShiftNetV2(in_channels=3, num_classes=10)
    x = torch.randn(4, 3, 64, 64)
    y = model(x, verbose=True)
    print("\nFinal:", y.shape)
