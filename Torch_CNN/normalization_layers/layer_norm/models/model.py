import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta  = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # x: (N, C, H, W)
        mean = x.mean(dim=1, keepdim=True)                
        var  = (x - mean).pow(2).mean(dim=1, keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return xhat * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)


class SimpleLN_CNN(nn.Module):
    """
    Basit LayerNorm'lu CNN (görüntü sınıflandırma örneği)
    Giriş:  (N, 3, H, W)
    Çıkış:  (N, num_classes)
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            LayerNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            LayerNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)      
        return self.fc(x)

if __name__ == "__main__":
    model = SimpleLN_CNN(num_classes=10)
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    print(out.shape)  