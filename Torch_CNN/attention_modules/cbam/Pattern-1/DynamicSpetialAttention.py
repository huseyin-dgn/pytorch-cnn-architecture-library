import torch
import torch.nn as nn
import torch.nn.functional as F

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
                k += 1
            if k<1:
                raise ValueError("k en az 1 olmalı")
            ks.append(k)

        self.branches = nn.ModuleList()
        for k in ks:
            p = k//2
            self.branches.append(nn.Conv2d(2,1,kernel_size = k , padding = p , bias = False))

        if use_dilated:
            k = int(dilated_kernel)
            if k % 2 == 0 :
                k +=1
            if dilated_d < 1:
                raise ValueError("Dilated_d en az 1 olmalı")
        
            p = dilated_d * (k-1) // 2
            self.branches.append(nn.Conv2d(2,1,kernel_size = k , dilation = dilated_d,padding = p,bias =False))
        
        self.num_branches = len(self.branches)

        if gate.lower() == "sigmoid":
            self.gate_fn = torch.sigmoid
        else:
            self.gate_fn = F.hardsigmoid

        self.learnable_temperature = learnable_temperature
        if learnable_temperature:
            t_raw = torch.tensor(float(temperature))
            t_inv = torch.log(torch.exp(t_raw - 1.0) + eps)
            self.t_raw = nn.Parameter(t_inv)
        else:
            self.register_buffer("T",torch.tensor(float(temperature)))

        self.router = nn.Sequantial(
            nn.AdaptivePool2d(1),
            nn.Conv2d(2,router_hidden,1,bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(router_hidden,self.num_branches,1,bias=bias))
    
    def get_T(self) -> torch.Tensor:
        if self.learnable_temperature:
            return F.softplus(self.t_raw) + self.eps
        return self.T
    
    def forward(self,x:torch.Tensor):
        avg_map = torch.mean(x,dim=1,keepdim = True)
        max_map,_ = torch.max(x,dim=1,keepdim=True)
        s = torch.stack([avg_map,max_map],dim=1)

        logits = self.router(s).flatten(1)
        rw = torch.softmax(logits,dim=1)

        z = torch.stack([br(s) for br in self.branches],dim=1)
        wlogit = (rw[:,:,None,None,None]*z).sum(dim=1)

        T = self.get_T()
        sa = self.gate_fn(wlogit/T)
        y = x*sa

        if self.return_router_weights:
            return y,sa,rw
        return y,sa