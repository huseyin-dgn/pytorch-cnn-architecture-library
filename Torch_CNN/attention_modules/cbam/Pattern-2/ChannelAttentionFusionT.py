import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus_inverse(y:torch.Tensor , eps:float = 1e-6) -> torch.Tensor:
    return torch.log(torch.exp(y) - 1.0 , min = eps)

def get_gate(gate:str):
    g = gate.lower()
    if g =="sigmoid":
        return torch.sigmoid
    if g == "hardsigmoid":
        return F.hardsigmoid
    raise ValueError("gate sigmoid yada hardsigmoid olmalı")

def get_act(act:str):
    a = act.lower()
    if a == "relu":
        return nn.ReLU(inplace = True)
    if a == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError("act relu veya silu olmalı")

class ChannelAttentionFusionT(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        min_hidden: int = 4,
        fusion: str = "softmax",        
        gate: str = "sigmoid",          
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        eps: float = 1e-6,
        act: str = "relu",
        bias: bool = True,
        fusion_router_hidden: int = 16,   
        return_fusion_weights: bool = False,
    ):
        super().__init__()

        if fusion not in ("sum", "softmax"):
            raise ValueError("fusion 'sum' veya 'softmax' olmalı.")
        if temperature <= 0:
            raise ValueError("temperature pozitif olmalı.")
        if fusion_router_hidden < 1:
            raise ValueError("fusion_router_hidden >= 1 olmalı.")
        
        self.eps = float(eps)
        self.fusion = fusion
        self.return_fusion_weights = bool(return_fusion_weights)

        self.gate_fn = get_gate(gate)

        hidden = max(min_hidden,channels//reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channels,hidden,1,bias = bias)
        self.act = get_act(act)
        self.fc2 = nn.Conv2d(hidden,channels,1,bias=bias)

        if self.fusion == "softmax":
            self.fusion_router = nn.Sequential(
                nn.Conv2d(2*channels,fusion_router_hidden,1,bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(fusion_router_hidden , 2 , 1 , bias=True),
            )
        else:
            self.fusion_router = None

        self.learnable_temperature = learnable_temperature
        if learnable_temperature:
            t0 = torch.tensor(float(temperature))
            t_inv = softplus_inverse(t0,eps=self.eps)
            self.t_raw = nn.Parameter(t_inv)
        else:
            self.register_buffer("T",torch.tensor(float(temperature)))

    def get_T(self) -> torch.Tensor:
        if self.learnable_temperature:
            return F.softplus(self.t_raw) + self.eps
        return self.T
    
    def mlp(self,s:torch.Tensor):
        return self.fc2(self.act(self.fc1(s)))
    
    def forward(self,x:torch.Tensor):
        avg_s = self.avg_pool(x)
        max_s = self.max_pool(x)

        a = self.mlp(avg_s)
        m = self.mlp(max_s)

        fusion_w = None
        if self.fuison == "sum":
            z = a+m
        else:
            s_cat = torch.cat([avg_s,max_s],dim=1)
            logits = self.fusion_router(s_cat).flatten(1)
            fusion_w = torch.softmax(logits,dim=1)
            z = fusion_w[:,0].view(-1,1,1,1) * a + fusion_w[:,1].view(-1,1,1,1) * m

        T = self.get_T()
        ca = self.gate_fn(z/T)
        y = x * ca

        if self.return_fusion_weights and ( fusion_w is not None):
            return y , ca , fusion_w
        return y, ca