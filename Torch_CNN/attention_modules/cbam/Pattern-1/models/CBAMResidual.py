import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return_fusion_weights: bool = False,
    ):
        super().__init__()

        if fusion not in ("sum", "softmax"):
            raise ValueError("fusion 'sum' veya 'softmax' olmalı.")
        if gate.lower() not in ("sigmoid", "hardsigmoid"):
            raise ValueError("gate 'sigmoid' veya 'hardsigmoid' olmalı.")
        if temperature <= 0:
            raise ValueError("temperature pozitif olmalı.")
        if act not in ("relu", "silu"):
            raise ValueError("act 'relu' veya 'silu' olmalı.")
        
        self.eps = eps
        self.fusion = fusion
        self.return_fusion_weights = return_fusion_weights

        hidden = max(min_hidden , channels//reduction)
        self.avg_pool = nn.AdaptivePool2d(1)
        self.max_pool = nn.AdaptivePool2d(1)
        self.fc1 = nn.Conv2d(channels,hidden,1,bias=bias)
        self.act = nn.ReLU(inplace=True) if act =="relu" else nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden,channels,1,bias = bias)
        
        if gate.lower() =="sigmoid":
            self.gate_fn = torch.sigmoid
        else:
            self.gate_fn = F.hardsigmoid

        if fusion == "softmax":
            self.fusion_logits = nn.Parameter(torch.zeros(2))
        else:
            self.fusion_logits = None

        self.learnable_temperature = learnable_temperature
        if learnable_temperature:
            t_raw = torch.tensor(float(temperature))
            t_inv = torch.log(torch.exp(t_raw)-1.0) + eps
            self.t_raw = nn.Parameter(t_inv)
        else:
            self.register_buffer("T",torch.tensor(float(temperature)))

    def get_T(self) -> torch.Tensor:
        if self.learnable_temperature:
            return F.softplus(self.t_raw) + self.esp
        return self.t_raw
    
    def mlp(self, s:torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(s)))

    def forward(self,x:torch.Tensor):
        a = self.mlp(self.avg_pool(x))
        m = self.mlp(self.max_pool(x))

        fusion_w = None
        if self.fusion == "sum":
            z = a+m
        else:
            fusion_w = torch.softmax(self.fusion_logits,dim=0)
            z = fusion_w[0] * a + fusion_w[1]*m

        T = self.get_T()
        ca = self.gate_fn(z/T)
        y = x*ca

        if self.return_fusion_weights and (fusion_w is not None):
            return y , ca , fusion_w
    
        return y , ca

class DynamicSpatialAttention(nn.Module):
    def __init__(
        self,
        kernels=(3, 7),
        use_dilated: bool = True,
        dilated_kernel: int = 7,
        dilated_d: int = 2,
        gate: str = "sigmoid",      
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        eps: float = 1e-6,
        router_hidden: int = 8,
        bias: bool = True,
        return_router_weights: bool = False,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature pozitif olmalı.")
        if gate.lower() not in ("sigmoid", "hardsigmoid"):
            raise ValueError("gate 'sigmoid' veya 'hardsigmoid' olmalı.")
        if router_hidden < 1:
            raise ValueError("router_hidden >= 1 olmalı.")

        self.eps = eps
        self.return_router_weights = return_router_weights

        ks = []
        for k in kernels:
            k = int(k)
            if k % 2 == 0:
                k+=1
            if k<1:
                raise ValueError("K 1 den büyük olmalıdır...")
            ks.append(k)

        self.branches = nn.ModuleList()
        for k in ks:
            p = k//2
            self.branches.apppend(nn.Conv2d(2,1,kernel_size = k , padding = p , bias = bias))
        
        if use_dilated:
            k = int(dilated_kernel)
            if k % 2 == 0:
                k =+1
            if dilated_d < 1:
                raise ValueError("dilated değeri 1 den büyük olmalı...")
            p = dilated_d * (k-1) //2
            self.branches.append(nn.Conv2d(2,1,kernel_size = k , padding = p , bias = bias))
        
        self.num_branches = len(self.branches)

        if gate.lower() == "sigmoid":
            self.gate_dn = torch.sigmoid
        else:
            self.gate_dn = F.hardsigmoid
        
        self.learnable_temperature = learnable_temperature
        if learnable_temperature:
            t_raw = torch.tensor(float(temperature))
            t_inv = torch.log(torch.exp(t_raw) - 1.0 + eps)
            self.t_raw = nn.Parameter(t_inv)
        else:
            self.register_buffer("T",torch.tensor(float(temperature)))
        
        self.router = nn.Sequential(
            nn.AdaptivePool2d(1),
            nn.Conv2d(2,router_hidden,1,bias = bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(router_hidden,self.num_branches,1,bias=bias))
    
    def get_T(self) -> torch.Tensor:
        if self.learnable_temperature:
            return F.softplus(self.t_raw) + self.eps
        return self.T
    
    def forward(self,x:torch.Tensor):
        avg_map = torch.mean(x,dim=1,keepdim=True)
        max_map , _ = torch.max(x,dim=1,keepdim=True)
        s = torch.cat([avg_map,max_map],dim=1)

        logits = self.router(s).flatten(1)
        rw = torch.softmax(logits,dim=1)

        z = torch.stack([br(s) for br in self.branches],dim=1)
        wlogits = (rw[:,:,None,None,None]*z).sum(dim=1)

        T = self.get_T()
        sa = self.gate_dn(wlogits / T)
        y = x*sa

        if self.return_router_weights:
            return y,sa,rw
        return y,sa

class CBAMResidualDynamicSA(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        min_hidden: int = 4,
        ca_fusion: str = "softmax",
        ca_gate: str = "sigmoid",
        ca_temperature: float = 1.0,
        ca_act: str = "relu",
        sa_gate: str = "sigmoid",
        sa_temperature: float = 1.0,
        learnable_temperature: bool = False,
        sa_kernels=(3, 7),
        sa_use_dilated: bool = True,
        sa_dilated_kernel: int = 7,
        sa_dilated_d: int = 2,
        sa_router_hidden: int = 8,
        residual: bool = True,
        alpha_init: float = 1.0,
        learnable_alpha: bool = False,
        return_maps: bool = False,
    ):
        super().__init__()

        self.return_maps = return_maps
        self.residual = residual

        self.ca = ChannelAttentionFusionT(
            channels=channels,
            reduction=reduction,
            min_hidden=min_hidden,
            fusion=ca_fusion,
            gate = ca_gate,
            temperature=ca_temperature,
            learnable_temperature=learnable_temperature,
            act = ca_act,
            return_fusion_weights= return_maps)

        self.sa = DynamicSpatialAttention(
            kernels=sa_kernels,
            dilated_d=sa_use_dilated,
            dilated_d = sa_dilated_d,
            dilated_kernel=sa_dilated_kernel,
            gate=sa_gate,
            temperature=sa_temperature,
            learnable_temperature=learnable_temperature,
            router_hidden=sa_router_hidden,
            return_router_weights=return_maps)
        
        if residual:
            if learnable_alpha:
                self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
            else:
                self.register_buffer("alpha",torch.tensor(float(alpha_init)))

    def forward(self,x:torch.Tensor):
        if self.return_maps:
            y , ca , fusion_w = self.ca(x)
            y , sa , router_w = self.sa(y)
            out = x + self.alpha * (y-x) if self.residual else y
            return out , ca,sa,fusion_w,router_w

        y,_ = self.ca(x)
        y,_ = self.sa(y)
        out = x + self.alpha * (y-x) if self.residual else y
        return out 