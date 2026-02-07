import torch
import torch.nn as nn
import torch.nn.functional as F

class WideResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout_p: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)

        self.dropout = nn.Dropout(p=dropout_p) if dropout_p and dropout_p > 0 else None

        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):

    #### F(x) = Conv2(ReLU(BN2( Conv1(ReLU(BN1(x))) ))) ####

        skip = x if self.proj is None else self.proj(x)

        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        out = self.conv1(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)

        return out + skip


def make_wide_layer(in_ch: int, out_ch: int, num_blocks: int, stride: int, dropout_p: float):
    layers = [WideResidualBlock(in_ch, out_ch, stride=stride, dropout_p=dropout_p)]
    for _ in range(1, num_blocks):
        layers.append(WideResidualBlock(out_ch, out_ch, stride=1, dropout_p=dropout_p))
    return nn.Sequential(*layers)


class WideResNet(nn.Module):
                                # ``` # 
        # Wide residual, kanalların artırılarak ilerletildiği bir mimari yaklaşımıdır.
        # k widen factor, stage bazında kanal sayısını çarpar.
        # n her stage’de kaç residual blok olduğunu belirler.
        # Genişletme yalnızca residual bağlantıya değil,
        # residual bloğun içindeki tüm konvolüsyonlara uygulanır.
        # Model, residual bloklardan oluştuğu için “wide residual” denir.
                                # ``` # 

    def __init__(self, num_classes: int = 10, depth: int = 28, widen_factor: int = 2,
                 dropout_p: float = 0.0, in_channels: int = 3):
        # depth = Ağın toplam katman derinliği demektir :: 6n+4 
        # widen_factor yani k ise kanalları büyütüyor.Burası wide kısmı.
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth 6n+4 olmalı (örn 28)."
        n = (depth - 4) // 6  

        base = 16
        ch1 = base * widen_factor
        ch2 = base * 2 * widen_factor
        ch3 = base * 4 * widen_factor

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        self.stage1 = make_wide_layer(base, ch1, num_blocks=n, stride=1, dropout_p=dropout_p)
        self.stage2 = make_wide_layer(ch1,  ch2, num_blocks=n, stride=2, dropout_p=dropout_p)
        self.stage3 = make_wide_layer(ch2,  ch3, num_blocks=n, stride=2, dropout_p=dropout_p)

        # Final
        self.bn = nn.BatchNorm2d(ch3)
        self.pool = nn.AdaptiveAvgPool2d(1) # H×W ne olursa olsun 1×1’e indirir
        self.fc = nn.Linear(ch3, num_classes)

    def forward(self, x):
        x = self.stem(x) # Modelin giriş katmanı
        x = self.stage1(x) # ANA GÖVDE
        x = self.stage2(x) # ANA GÖVDE
        x = self.stage3(x) # ANA GÖVDE

        x = self.bn(x)
        x = F.relu(x, inplace=True) # final clean-up

        x = self.pool(x).flatten(1) 
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    model = WideResNet(num_classes=10, depth=28, widen_factor=2, dropout_p=0.0)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print("logits:", y.shape)  # [4, 10]