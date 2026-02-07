import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Routing (Gating) ağı
# ---------------------------
class RoutingMLP(nn.Module):
    def __init__(self, cin: int, K: int = 4, reduction: int = 4, temperature: float = 1.0):
        super().__init__()
        hidden = max(1, cin // reduction)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(cin, hidden)
        self.fc2 = nn.Linear(hidden, K)
        self.temperature = temperature

    def forward(self, x):
        v = self.gap(x).flatten(1)                 # (B,C)
        h = F.relu(self.fc1(v))                    # (B,hidden)
        logits = self.fc2(h)                       # (B,K)
        a = F.softmax(logits / self.temperature, dim=1)
        return a                                   # (B,K)

class RepVGGBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, deploy=False):
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.stride = stride
        self.deploy = deploy
        self.act = nn.ReLU(inplace=True)

        if deploy:
            self.rbr_reparam = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=True)
        else:
            self.rbr_3x3 = nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout)
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(cout)
            )
            if cout == cin and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(cout)
            else:
                self.rbr_identity = None

    def forward(self, x):
        if self.deploy:
            return self.act(self.rbr_reparam(x))

        out = self.rbr_3x3(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        return self.act(out)

    def get_equivalent_kernel_bias(self):
        conv3, bn3 = self.rbr_3x3[0], self.rbr_3x3[1]
        W3, b3 = fuse_conv_bn(conv3, bn3)

        conv1, bn1 = self.rbr_1x1[0], self.rbr_1x1[1]
        W1, b1 = fuse_conv_bn(conv1, bn1)
        W1 = pad_1x1_to_3x3(W1)

        if self.rbr_identity is not None:
            Wid, bid = fuse_identity_bn(self.cout, self.rbr_identity, device=W3.device, dtype=W3.dtype)
        else:
            Wid = torch.zeros_like(W3)
            bid = torch.zeros_like(b3)

        W_eq = W3 + W1 + Wid
        b_eq = b3 + b1 + bid
        return W_eq, b_eq

    def switch_to_deploy(self):
        if self.deploy:
            return
        W, b = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(self.cin, self.cout, 3, stride=self.stride, padding=1, bias=True)
        self.rbr_reparam.weight.data = W
        self.rbr_reparam.bias.data = b

        del self.rbr_3x3
        del self.rbr_1x1
        if hasattr(self, 'rbr_identity'):
            del self.rbr_identity
        self.deploy = True


# ---------------------------
# Dynamic Conv2d (okunur demo versiyon)
# ---------------------------
class DynamicConv2d(nn.Module):
    def __init__(self, cin, cout, k=3, stride=1, padding=1,
                 K=4, reduction=4, temperature=1.0, bias=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.K = K

        # Kernel bankası: (K, cout, cin, k, k)
        self.weight = nn.Parameter(torch.randn(K, cout, cin, k, k) * 0.02)

        # (Opsiyonel) bias bankası: (K, cout)
        self.bias_bank = nn.Parameter(torch.zeros(K, cout)) if bias else None

        self.router = RoutingMLP(cin=cin, K=K, reduction=reduction, temperature=temperature)

    def forward(self, x, return_routing=False):
        B = x.size(0)
        a = self.router(x)  # (B,K)

        # Wdyn: (B, cout, cin, k, k)
        Wdyn = torch.einsum("bk,kocij->bocij", a, self.weight)

        bdyn = None
        if self.bias_bank is not None:
            bdyn = torch.einsum("bk,kc->bc", a, self.bias_bank)  # (B, cout)

        outs = []
        for i in range(B):
            yi = F.conv2d(
                x[i:i+1],
                Wdyn[i],
                bias=None if bdyn is None else bdyn[i],
                stride=self.stride,
                padding=self.padding
            )
            outs.append(yi)

        y = torch.cat(outs, dim=0)
        return (y, a) if return_routing else y

# ---------------------------
# Dynamic Conv Block
# ---------------------------
class DynamicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, K=4, reduction=4, temperature=1.0):
        super().__init__()
        self.dyn = DynamicConv2d(cin, cout, k=3, stride=stride, padding=1,
                                 K=K, reduction=reduction, temperature=temperature, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, return_routing=False):
        y, a = self.dyn(x, return_routing=True)
        y = self.act(self.bn(y))
        return (y, a) if return_routing else y

# ---------------------------
# Tam Model: DynamicCNN
# ---------------------------
class DynamicCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, K=4, temperature=1.0):
        super().__init__()

        # Stem (normal conv)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Dynamic stages
        self.s1 = DynamicBlock(32, 64,  stride=2, K=K, reduction=4, temperature=temperature)   # 64->32
        self.s2 = DynamicBlock(64, 128, stride=2, K=K, reduction=4, temperature=temperature)   # 32->16
        self.s3 = DynamicBlock(128,256, stride=2, K=K, reduction=4, temperature=temperature)   # 16->8

        # Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, verbose=False, return_routing=False):
        if verbose: print("input:", x.shape)

        x = self.stem(x)
        if verbose: print("stem :", x.shape)

        x, a1 = self.s1(x, return_routing=True)
        if verbose: print("s1   :", x.shape, " routing:", a1.shape)

        x, a2 = self.s2(x, return_routing=True)
        if verbose: print("s2   :", x.shape, " routing:", a2.shape)

        x, a3 = self.s3(x, return_routing=True)
        if verbose: print("s3   :", x.shape, " routing:", a3.shape)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        if verbose: print("out  :", out.shape)

        if return_routing:
            return out, {"s1": a1, "s2": a2, "s3": a3}
        return out

# ---------------------------
# Çalıştırma örneği
# ---------------------------
if __name__ == "__main__":
    model = DynamicCNN(in_channels=3, num_classes=10, K=4, temperature=1.0)
    x = torch.randn(2, 3, 64, 64)

    logits, routing = model(x, verbose=True, return_routing=True)

    print("\nlogits:", logits.shape)
    print("routing s1 row sums:", routing["s1"].sum(dim=1))
    print("routing s2 row sums:", routing["s2"].sum(dim=1))
    print("routing s3 row sums:", routing["s3"].sum(dim=1))
