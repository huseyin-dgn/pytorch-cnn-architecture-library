import torch
import torch.nn as nn
import torch.nn.functional as F


def make_odd(k:int) -> int:
    k = int(k)
    if k < 1:
        raise ValueError("Kernel size 1 den büyük olmalı")
    return k if (k % 2 == 1) else (k+1)

def sotfplus_inverse(y:torch.Tensor , eps:float = 1e-6) -> torch.Tensor:
    return torch.log(torch.clamp(torch.ecp(y)-1.0 , min = eps))

def get_gate(gate:str):
    g = gate.lower()
    if g == "sigmoid":
        return torch.sigmoid
    if g == "hardsigmoid":
        return F.hardsigmoid
    raise ValueError("gate ya sigmoid yada hardsigmoid olmalı...")

class DWPointwiseBranches(nn.Module):
    def __init__(self, in_ch:int , k:int , dilation:int = 1):
        super().__init__()
        k = make_odd(k)
        dilation = int(dilation)
        if dilation < 1:
            raise ValueError("Dilation 1 veya 1 den büyük olmalıdır...")
        pad = dilation * (k-1)//2

        self.dw = nn.Conv2d(in_ch , in_ch , kernel_size = k , padding = pad , dilation = dilation , groups = in_ch , bias=False)
        self.pw = nn.Conv2d(in_ch,1,kernel_size = 1 , bias=True)
    
    def forward(self,s:torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(s))
    
class DynamicSpatialAttention(nn.Module):
    def __init__(self,kernels=(3,7), use_dilated:bool = True , dilated_kernel:int = 7 , dilated_d : int = 2 , gate:str = "sigmoid",temperature:float = 1.0 ,
                 learnable_temperature:bool = True , eps:float = 1e-6 , router_hidden:int = 8 , bias:bool = True, return_router_weights : bool = False , coord_norm : str = "minus1to1"):
        super().__init__()    

        if temperature <= 0:
            raise ValueError("temperature pozitif olmalı.")
        if router_hidden < 1:
            raise ValueError("router_hidden >= 1 olmalı.")
        if coord_norm not in ("minus1to1", "0to1"):
            raise ValueError("coord_norm 'minus1to1' veya '0to1' olmalı.") 
        
        self.eps = float(eps)
        self.return_router_weights = bool(return_router_weights)
        self.gate_fn = get_gate(gate)
        self.coord_norm = coord_norm
        
        in_ch = 4

        ks = []
        for k in kernels:
            ks.append(make_odd(int(k)))
        
        branches = []
        for k in ks:
            branches.append(DWPointwiseBranches(in_ch=in_ch,k=k,dilation=1))
        
        if use_dilated:
            dk = make_odd(int(dilated_kernel))
            dd = int(dilated_d)
            if dd < 1:
                raise ValueError("Dilated_d 1 veya 1 den büyük olmalı...")
            branches.append(DWPointwiseBranches(in_ch=in_ch,k = dk , dilation=dd))
        
        self.branches = branches
        self.num_branches = len(self.branches)

        self.router = nn.Sequantial(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch , router_hidden , kernel_size = 1 ,  bias = bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(router_hidden,self.num_branches,kernel_size = 1 , bias = bias),)
        
        self.learnable_temperature = bool(learnable_temperature)
        if self.learnable_temperature:
            t0 = torch.tensor(float(temperature))
            t_inv = sotfplus_inverse(t0 , eps=self.eps)
            self.t_raw = nn.Parameter(t_inv)
        else:
            self.register_buffer("T",torch.tensor(float(temperature)))

        self.coord_cache = {}

    def get_T(self)->torch.Tensor:
        if self.learnable_temperature:
            return F.softplus(self.t_raw) + self.eps
        return self.T
    
    def coords(self,B:int,H:int , W:int , device , dtype):
        key = (H,W,str(device),str(dtype),self.coord_norm)

        if key in self.coord_cache:
            xg,yg = self.coord_cache[key]
        else:
            if self.coord_norm == "minus1to1":
                xs = torch.linspace(-1.0,1.0,W,device=device,dtype=dtype)
                ys = torch.linspace(-1.0,1.0,H,device=device,dtype=dtype)
            else:
                xs = torch.linspace(0.0 ,1.0 , W,device=device,dtype=dtype)
                ys = torch.linspace(0.0,1.0 , H ,device=device,dtype=dtype)

            yy , xx = torch.meshgrid(ys,xs,indexing="ij")

            xg  = xx.unsqueeze(0).unsqueeze(0)
            yg = yy.unsqueeze(0).unsqueeze(0)
            self.coord_norm[key] = (xg,yg)
        return xg.expand(B,-1,-1,-1) , yg.expand(B,-1,-1,-1)
    
    def forward(self,x:torch.Tensor):
        B,C,H,W = x.shape

        avg_max = torch.mean(x,dim=1,keepdim=True)
        max_map , _ = torch.max(x,dim=1,keepdim=True)

        x_coord , y_coord = self.coords(B,H,W,x.device,y.device)
        s = torch.cat([avg_max,max_map,x_coord,y_coord],dim=1)

        logits = self.router(s).flatten(1)
        rw = torch.softmax(logits,dim=1)

        z = torch.stack([br(s) for br in self.branches],dim=1)
        wlogits= (rw[:,:, None,None,None]*z).sum(dim=1)

        T = self.get_T()
        sa = self.gate_fn(z/T)
        Y = x*sa

        if self.return_router_weights:
            return Y , sa , rw
        return Y,sa 