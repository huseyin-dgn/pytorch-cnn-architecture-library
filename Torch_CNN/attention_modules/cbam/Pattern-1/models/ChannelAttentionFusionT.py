# Öncelikle gerekli parametreleri aldık ve fusion, return_fusion_weights ve eps 
# değişkenlerini self içine kaydettik. Daha sonra hidden kanal sayısını belirledik;
# bunu yaparken channels // reduction değerinin çok küçük olmaması için
# max(min_hidden, ...) kullandık. Ardından kanal bilgisini sıkıştırmak için
# AdaptiveAvgPool2d(1) ve AdaptiveMaxPool2d(1) tanımladık. Sonrasında bu 
# sıkıştırılmış kanal bilgilerini işleyecek olan MLP yapısını kurduk;
# bunun için önce channels → hidden dönüşümünü yapan fc1 katmanını, 
# ardından aktivasyon fonksiyonunu (ReLU veya SiLU) ve son olarak 
# hidden → channels dönüşümünü yapan fc2 katmanını tanımladık. 
# Daha sonra kanal maskesini üretirken kullanılacak gate_fn fonksiyonunu 
# (sigmoid veya hardsigmoid) belirledik. Sıcaklığın öğrenilebilir 
# olup olmadığına göre iki yol izledik; eğer öğrenilebilir ise verilen 
# temperature değerini başlangıçta korunacak şekilde softplus’ın tersini
# alarak t_inv hesapladık ve bunu t_raw adlı öğrenilebilir parametre olarak 
# tanımladık, eğer öğrenilebilir değilse temperature değerini register_buffer 
# ile sabit bir değişken olarak modele ekledik. Ardından sıcaklığı güvenli 
# şekilde elde etmek için _get_T fonksiyonunu tanımladık ve kanal özetlerini
# işlemek için _mlp yardımcı fonksiyonunu yazdık. Forward aşamasında önce girişten
# avg ve max pooling ile kanal özetlerini çıkardık ve bunları MLP’den 
# geçirerek a ve m değerlerini elde ettik. Eğer fusion yöntemi sum ise bu
# iki değeri doğrudan topladık, değilse fusion_logits üzerinden softmax 
# alarak modelin avg ve max kaynaklarından hangisine daha fazla güveneceğini 
# öğrendiği ağırlıkları elde ettik ve bu ağırlıklarla a ve m değerlerini 
# birleştirerek z değerini oluşturduk. Daha sonra _get_T fonksiyonu ile 
# sıcaklık değerini aldık, z / T işlemini uygulayıp bunu gate_fn fonksiyonundan
# geçirerek kanal attention maskesini ca olarak ürettik ve son olarak bu maskeyi 
# giriş tensorü ile çarparak attention uygulanmış çıktı y değerini elde ettik; 
# eğer istenirse analiz amacıyla fusion ağırlıklarını da çıktı olarak döndürdük.


import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttentionFusionT(nn.Module):
    def __init__(self,channels:int,reductions:int=8,eps:float=1e-6,temperature:float = 1.0,min_hidden:int = 4, fusion :str = "softmax", gate:str ="sigmoid",learnable_temperature:bool = False,act:str="relu",bias:bool = True,return_fusion_weights:bool = False):
        super().__init__()
        if fusion not in ("sum", "softmax"):
            raise ValueError("fusion 'sum' veya 'softmax' olmalı.")
        if gate.lower() not in ("sigmoid", "hardsigmoid"):
            raise ValueError("gate 'sigmoid' veya 'hardsigmoid' olmalı.")
        if temperature <= 0:
            raise ValueError("temperature pozitif olmalı.")
        if act not in ("relu", "silu"):
            raise ValueError("act 'relu' veya 'silu' olmalı.")
        
        self.fusion = fusion
        self.return_fusion_weights = return_fusion_weights
        self.eps = eps

        hidden = max(min_hidden,channels//reductions)

        self.avg = nn.AdaptivePool2d(1)
        self.max = nn.AdaptivePool2d(1)

        self.fc1 = nn.Conv2d(channels,hidden,1,bias=bias)
        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.SİLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=bias)

        if gate.lower() == "sigmoid":
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
            t_inv = torch.log(torch.exp(t_raw) - 1.0 + eps)
            self.t_raw = nn.Parameter(t_inv)
        else:
            self.register_buffer("T",torch.tensor(float(temperature)))

# Biz T’yi eğitmek istiyoruz
# AMA T asla 0 veya negatif olmasın istiyoruz.

# Bunun için T’yi direkt eğitmiyoruz.
# Onun yerine başka bir sayı eğitiyoruz: t_raw.

# t_inv, softplus’tan geçince temperature versin diye,
# softplus formülünü geriye doğru çözerek elde edilen
# güvenli başlangıç değeridir.”

    def _get_T(self)->torch.Tensor:
        if self.learnable_temperature:
            return F.softplus(self.t_raw) + self.eps
        return self.T

    def _mlp(self,s:torch.Tensor) ->torch.Tensor:
        return self.fc2(self.act(self.fc1(s)))
    
    def forward(self,x:torch.Tensor):
        a = self._mlp(self.avg(x))
        m = self._mlp(self.max(x))

        fusion_w = None
        if self.fusion == "sum":
            z = a+m
        else:

# Bunun anlamı şu:

# “Avg mi daha güvenilir, Max mi daha güvenilir? Bunu model öğrensin.”

# Bazı dataset/backbone’larda avg daha stabil olur → model w0’ı büyütür.

# Bazılarında max daha iyi sinyal verir → model w1’i büyütür.

# Softmax kullanınca w0+w1=1 olur, yani karışım oranı gibi davranır.
            fusion_w = torch.softmax(self.fusion_logits,dim = 0)
            z = fusion_w[0] * a + fusion_w[1] * m
        
        T = self._get_T()
        ca = self.gate_fn(z/T)
        y = ca * x
        
        if self.return_fusion_weights and ( fusion_w is not None):
            return y, ca , fusion_w
        return y , ca