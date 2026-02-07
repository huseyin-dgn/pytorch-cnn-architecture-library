import torch
import torch.nn as nn


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


class RepVGGClassifier(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, deploy=False):
        super().__init__()
        self.deploy = deploy

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            RepVGGBlock(32, 64,  stride=2, deploy=deploy),
            RepVGGBlock(64, 64,  stride=1, deploy=deploy),
        )
        self.stage2 = nn.Sequential(
            RepVGGBlock(64, 128, stride=2, deploy=deploy),
            RepVGGBlock(128,128, stride=1, deploy=deploy),
        )
        self.stage3 = nn.Sequential(
            RepVGGBlock(128,256, stride=2, deploy=deploy),
            RepVGGBlock(256,256, stride=1, deploy=deploy),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, verbose=False):
        if verbose: print("input:", x.shape)
        x = self.stem(x)
        if verbose: print("stem :", x.shape)

        x = self.stage1(x)
        if verbose: print("s1   :", x.shape)

        x = self.stage2(x)
        if verbose: print("s2   :", x.shape)

        x = self.stage3(x)
        if verbose: print("s3   :", x.shape)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        if verbose: print("out  :", out.shape)
        return out

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()
        self.deploy = True


if __name__ == "__main__":
    model = RepVGGClassifier(in_channels=3, num_classes=10, deploy=False)
    x = torch.randn(2, 3, 64, 64)

    model.eval()
    with torch.no_grad():
        y_train = model(x, verbose=True)

    model.switch_to_deploy()
    model.eval()
    with torch.no_grad():
        y_deploy = model(x, verbose=True)

    print("max_abs_diff:", (y_train - y_deploy).abs().max().item())
