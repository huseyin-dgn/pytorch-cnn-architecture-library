import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

def softplus_inverse(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(torch.clamp(torch.exp(y) - 1.0, min=eps))

def _get_gate(gate: str):
    g = gate.lower()
    if g == "sigmoid":
        return torch.sigmoid
    if g == "hardsigmoid":
        return F.hardsigmoid
    raise ValueError("gate 'sigmoid' veya 'hardsigmoid' olmalı.")

def _get_act(act: str):
    a = act.lower()
    if a == "relu":
        return nn.ReLU(inplace=True)
    if a == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError("act 'relu' veya 'silu' olmalı.")

class ChannelAttentionFusionT(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        min_hidden: int = 4,
        fusion: str = "softmax",
        gate: str = "sigmoid",
        temperature: float = 0.9,
        learnable_temperature: bool = False,
        eps: float = 1e-6,
        act: str = "relu",
        bias: bool = True,
        fusion_router_hidden: int = 16,
        return_fusion_weights: bool = False,
        t_min: float = 0.5,
        t_max: float = 3.0,
        router_temperature: float = 1.5,
        beta_ca: float = 0.35,
    ):
        super().__init__()

        if channels < 1:
            raise ValueError("channels >= 1 olmalı.")
        if reduction < 1:
            raise ValueError("reduction >= 1 olmalı.")
        if fusion not in ("sum", "softmax"):
            raise ValueError("fusion 'sum' veya 'softmax' olmalı.")
        if temperature <= 0:
            raise ValueError("temperature pozitif olmalı.")
        if fusion == "softmax" and fusion_router_hidden < 1:
            raise ValueError("fusion_router_hidden >= 1 olmalı.")
        if t_min <= 0 or t_max <= 0 or t_min > t_max:
            raise ValueError("T clamp aralığı hatalı.")
        if router_temperature <= 0:
            raise ValueError("router_temperature pozitif olmalı.")
        if beta_ca < 0:
            raise ValueError("beta_ca >= 0 olmalı.")

        self.eps = float(eps)
        self.fusion = fusion
        self.return_fusion_weights = bool(return_fusion_weights)
        self.gate_fn = _get_gate(gate)

        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.Tr = float(router_temperature)
        self.beta_ca = float(beta_ca)

        hidden = max(int(min_hidden), int(channels) // int(reduction))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=bias)
        self.act = _get_act(act) # silu , relu
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=bias)

        if self.fusion == "softmax":
            self.fusion_router = nn.Sequential(
                nn.Conv2d(2 * channels, fusion_router_hidden, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(fusion_router_hidden, 2, kernel_size=1, bias=True),
            )
            last = self.fusion_router[-1] # [-1] = “en sondaki eleman" 
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        else:
            self.fusion_router = None

        self.learnable_temperature = bool(learnable_temperature)
        if self.learnable_temperature:
            t0 = float(temperature)
            lo = self.t_min + self.eps
            hi = self.t_max - self.eps
            if lo >= hi:
                lo = self.t_min
                hi = self.t_max
            t0 = min(max(t0, lo), hi)
            t_inv = softplus_inverse(torch.tensor(t0), eps=self.eps)
            self.t_raw = nn.Parameter(t_inv)
        else:
            self.register_buffer("T", torch.tensor(float(temperature)))

    def get_T(self, x: torch.Tensor) -> torch.Tensor:
        if self.learnable_temperature:
            T = F.softplus(self.t_raw) + self.eps
        else:
            T = self.T
        T = T.to(device=x.device, dtype=x.dtype)
        return T.clamp(self.t_min, self.t_max)

    def mlp(self, s: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(s)))

    def forward(self, x: torch.Tensor):
        avg_s = self.avg_pool(x)
        max_s = self.max_pool(x)

        a = self.mlp(avg_s)
        m = self.mlp(max_s)

        fusion_w = None
        if self.fusion == "sum":
            z = a + m
        else:
            s_cat = torch.cat([avg_s, max_s], dim=1)
            logits = self.fusion_router(s_cat).flatten(1)
            fusion_w = torch.softmax(logits / self.Tr, dim=1)
            # Softmax: “bu sample’da avg mi daha önemli max mı?” sorusuna öğrenilen ağırlık
            # Tr :: :: Softmax’ın “keskinliğini” ayarlar
            w0 = fusion_w[:, 0].view(-1, 1, 1, 1)
            w1 = fusion_w[:, 1].view(-1, 1, 1, 1)
            z = w0 * a + w1 * m

        T = self.get_T(x)
        ca = self.gate_fn(z / T)
        # T gate’in agresifliğini kontrol eden “ısı” parametresi.

        scale_ca = 1.0 + self.beta_ca * (ca - 1.0)
        y = x * scale_ca

        if self.return_fusion_weights:
            return y, ca, fusion_w
        return y, ca


class HSwish(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * F.relu6(x + 3.0, inplace=True) / 6.0

def make_norm(norm: str, ch: int):
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm2d(ch)
    if norm == "gn":
        g = min(32, ch)
        while ch % g != 0 and g > 2:
            g //= 2
        if ch % g != 0:
            g = 2 if (ch % 2 == 0) else 1
        return nn.GroupNorm(g, ch)
    if norm == "none":
        return nn.Identity()
    raise ValueError("norm 'none', 'bn', 'gn' dışında olamaz.")

class CoordinateAttPlus(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction: int = 32,
        min_mid_channels: int = 8,
        act: str = "hswish",
        init_alpha: float = 0.7,
        learnable_alpha: bool = True,
        beta: float = 0.35,
        dilation: int = 2,
        norm: str = "gn",
        use_spatial_gate: bool = False,
        spatial_gate_beta: float = 0.35,
        scale_min: float = 0.6,
        scale_max: float = 1.6,
        head_init_std: float = 0.01,
    ):
        super().__init__()

        if in_channels < 1:
            raise ValueError("in_channels >= 1 olmalı.")
        if reduction < 1:
            raise ValueError("reduction >= 1 olmalı.")
        if dilation < 1:
            raise ValueError("dilation >= 1 olmalı.")
        if scale_min <= 0 or scale_max <= 0 or scale_min > scale_max:
            raise ValueError("scale clamp aralığı hatalı.")
        if head_init_std <= 0:
            raise ValueError("head_init_std pozitif olmalı.")
        if beta < 0:
            raise ValueError("beta >= 0 olmalı.")
        if spatial_gate_beta < 0:
            raise ValueError("spatial_gate_beta >= 0 olmalı.")

        self.beta = float(beta)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

        mid_floor = max(8, min(32, int(in_channels) // 4))
        mid = max(int(min_mid_channels), int(in_channels) // int(reduction))
        mid = max(mid, int(mid_floor))

        act_l = act.lower()
        if act_l == "hswish":
            self.act = HSwish()
        elif act_l == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act_l == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError("act 'hswish', 'relu', 'silu' olmalı.")

        self.shared_bottleneck_proj = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.shared_bottleneck_norm = make_norm(norm, mid)
        self.shared_bottleneck_refine = nn.Conv2d(mid, mid, 1, bias=False)
        self.shared_bottleneck_refine_norm = make_norm(norm, mid)

        self.h_local_dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0),
            groups=in_channels, bias=False
        )
        self.w_local_dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1),
            groups=in_channels, bias=False
        )

        d = int(dilation)
        self.h_dilated_dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=(3, 1), padding=(d, 0),
            dilation=(d, 1), groups=in_channels, bias=False
        )
        self.w_dilated_dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 3), padding=(0, d),
            dilation=(1, d), groups=in_channels, bias=False
        )

        self.h_channel_mixer = nn.Conv2d(in_channels, in_channels, 1, bias=True)
        self.w_channel_mixer = nn.Conv2d(in_channels, in_channels, 1, bias=True)

        self.h_attention_head = nn.Conv2d(mid, in_channels, 1, bias=True)
        self.w_attention_head = nn.Conv2d(mid, in_channels, 1, bias=True)

        nn.init.normal_(self.h_attention_head.weight, mean=0.0, std=float(head_init_std))
        nn.init.normal_(self.w_attention_head.weight, mean=0.0, std=float(head_init_std))
        if self.h_attention_head.bias is not None:
            nn.init.zeros_(self.h_attention_head.bias)
        if self.w_attention_head.bias is not None:
            nn.init.zeros_(self.w_attention_head.bias)

        eps = 1e-6
        a0 = float(init_alpha)
        a0 = min(max(a0, eps), 1.0 - eps)
        raw0 = torch.logit(torch.tensor(a0), eps=eps)
        # başlangıçta alpha=init_alpha olacak şekilde raw parametreyi ayarlamak”.

        if learnable_alpha:
            self.alpha_h_raw = nn.Parameter(raw0.clone())
            self.alpha_w_raw = nn.Parameter(raw0.clone())
        else:
            self.register_buffer("alpha_h_raw", raw0.clone())
            self.register_buffer("alpha_w_raw", raw0.clone())

        self.use_spatial_gate = bool(use_spatial_gate)
        self.spatial_gate_beta = float(spatial_gate_beta)
        if self.use_spatial_gate:
            self.spatial_gate_dw = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
            self.spatial_gate_pw = nn.Conv2d(in_channels, in_channels, 1, bias=True)

        self._last_ah = None
        self._last_aw = None

    def forward(self, x: torch.Tensor):
        _, _, H, W = x.shape

        h_profile = 0.5 * (x.mean(dim=3, keepdim=True) + x.amax(dim=3, keepdim=True))
        w_profile = 0.5 * (x.mean(dim=2, keepdim=True) + x.amax(dim=2, keepdim=True))

        h_ms = self.h_channel_mixer(self.h_local_dw(h_profile) + self.h_dilated_dw(h_profile))
        w_ms = self.w_channel_mixer(self.w_local_dw(w_profile) + self.w_dilated_dw(w_profile))
        w_ms = w_ms.permute(0, 1, 3, 2)

        hw = torch.cat([h_ms, w_ms], dim=2)

        mid = self.act(self.shared_bottleneck_norm(self.shared_bottleneck_proj(hw)))
        mid = self.act(self.shared_bottleneck_refine_norm(self.shared_bottleneck_refine(mid)))

        mid_h, mid_w = torch.split(mid, [H, W], dim=2)
        mid_w = mid_w.permute(0, 1, 3, 2)

        attn_h = F.hardsigmoid(self.h_attention_head(mid_h), inplace=False)
        attn_w = F.hardsigmoid(self.w_attention_head(mid_w), inplace=False)

        self._last_ah = attn_h.detach()
        self._last_aw = attn_w.detach()

        alpha_h = torch.sigmoid(self.alpha_h_raw).to(device=x.device, dtype=x.dtype)
        alpha_w = torch.sigmoid(self.alpha_w_raw).to(device=x.device, dtype=x.dtype)

        scale_h = (1.0 - alpha_h) + alpha_h * attn_h
        scale_w = (1.0 - alpha_w) + alpha_w * attn_w

        scale = scale_h * scale_w
        scale = 1.0 + self.beta * (scale - 1.0)
        scale = scale.clamp(self.scale_min, self.scale_max)
        out = x * scale

        if self.use_spatial_gate:
            sg = self.spatial_gate_pw(self.spatial_gate_dw(x))
            sg = F.hardsigmoid(sg, inplace=False)
            sg = 1.0 + self.spatial_gate_beta * (sg - 1.0)
            out = out * sg
        return out

    @torch.no_grad()
    def last_mask_stats(self):
        if (self._last_ah is None) or (self._last_aw is None):
            return None
        ah = self._last_ah
        aw = self._last_aw
        return {
            "a_h": {"min": float(ah.min()), "mean": float(ah.mean()), "max": float(ah.max()), "std": float(ah.std())},
            "a_w": {"min": float(aw.min()), "mean": float(aw.mean()), "max": float(aw.max()), "std": float(aw.std())},
        }


class CBAMChannelPlusCoord(nn.Module):
    def __init__(
        self,
        channels: int,
        ca_reduction: int = 16,
        ca_min_hidden: int = 4,
        ca_fusion: str = "softmax",
        ca_gate: str = "sigmoid",
        ca_temperature: float = 0.9,
        ca_act: str = "relu",
        ca_fusion_router_hidden: int = 16,
        learnable_temperature: bool = False,
        ca_t_min: float = 0.5,
        ca_t_max: float = 3.0,
        ca_router_temperature: float = 1.5,
        beta_ca: float = 0.35,
        coord_reduction: int = 32,
        coord_min_mid: int = 8,
        coord_act: str = "hswish",
        coord_init_alpha: float = 0.7,
        coord_learnable_alpha: bool = True,
        coord_beta: float = 0.35,
        coord_dilation: int = 2,
        coord_norm: str = "gn",
        coord_use_spatial_gate: bool = False,
        coord_spatial_gate_beta: float = 0.35,
        coord_scale_min: float = 0.6,
        coord_scale_max: float = 1.6,
        coord_head_init_std: float = 0.01,
        residual: bool = True,
        alpha_baslangic: float = 0.75,
        alpha_ogrenilsin: bool = False,
        monitor: bool = False,
        r_min: float = 0.45,
        ema_momentum: float = 0.95,
        min_kurtarma_orani: float = 0.2,
        alpha_etkin_min: float = 0.2,
        kurtarma_modu: str = "ratio_floor",
        return_maps: bool = False,
    ):
        super().__init__()
        if channels < 1:
            raise ValueError("channels >= 1 olmalı.")
        if not (0.0 < ema_momentum < 1.0):
            raise ValueError("ema_momentum (0,1) aralığında olmalı.")
        if r_min <= 0:
            raise ValueError("r_min pozitif olmalı.")
        if not (0.0 <= min_kurtarma_orani <= 1.0):
            raise ValueError("min_kurtarma_orani [0,1] aralığında olmalı.")
        if not (0.0 <= alpha_etkin_min <= 1.0):
            raise ValueError("alpha_etkin_min [0,1] aralığında olmalı.")
        if kurtarma_modu not in ("ratio_floor", "alpha_floor"):
            raise ValueError("kurtarma_modu 'ratio_floor' veya 'alpha_floor' olmalı.")

        self.return_maps = bool(return_maps)
        self.residual = bool(residual)

        self.monitor = bool(monitor)
        self.r_min = float(r_min)
        self.ema_m = float(ema_momentum)
        self.min_kurtarma_orani = float(min_kurtarma_orani)
        self.alpha_etkin_min = float(alpha_etkin_min)
        self.kurtarma_modu = str(kurtarma_modu)

        self.ca = ChannelAttentionFusionT(
            channels=channels,
            reduction=ca_reduction,
            min_hidden=ca_min_hidden,
            fusion=ca_fusion,
            gate=ca_gate,
            temperature=ca_temperature,
            learnable_temperature=learnable_temperature,
            eps=1e-6,
            act=ca_act,
            bias=True,
            fusion_router_hidden=ca_fusion_router_hidden,
            return_fusion_weights=self.return_maps,
            t_min=ca_t_min,
            t_max=ca_t_max,
            router_temperature=ca_router_temperature,
            beta_ca=beta_ca,
        )

        self.coord = CoordinateAttPlus(
            in_channels=channels,
            reduction=coord_reduction,
            min_mid_channels=coord_min_mid,
            act=coord_act,
            init_alpha=coord_init_alpha,
            learnable_alpha=coord_learnable_alpha,
            beta=coord_beta,
            dilation=coord_dilation,
            norm=coord_norm,
            use_spatial_gate=coord_use_spatial_gate,
            spatial_gate_beta=coord_spatial_gate_beta,
            scale_min=coord_scale_min,
            scale_max=coord_scale_max,
            head_init_std=coord_head_init_std,
        )

        if self.residual:
            # “alpha 0..1 aralığında garanti kalsın ama öğrenilebilir olsun.”
            eps = 1e-6
            a0 = float(alpha_baslangic)
            a0 = min(max(a0, eps), 1.0 - eps)
            raw0 = torch.logit(torch.tensor(a0), eps=eps)
            if alpha_ogrenilsin:
                self.alpha_ogrenilen_raw = nn.Parameter(raw0)
            else:
                self.register_buffer("alpha_ogrenilen_raw", raw0)

        self.register_buffer("r_ema", torch.tensor(1.0))

    def alpha_temiz(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.residual) or (not hasattr(self, "alpha_ogrenilen_raw")):
            return x.new_tensor(1.0)
        return torch.sigmoid(self.alpha_ogrenilen_raw).to(device=x.device, dtype=x.dtype)

    @staticmethod
    def std_batch_ort(x: torch.Tensor) -> torch.Tensor:
        return x.float().flatten(1).std(dim=1, unbiased=False).mean()
    # STD == “bu tensorde enerji var mı, dağılım canlı mı?

    @torch.no_grad()
    def r_ema_guncelle(self, r_out: torch.Tensor):
        r_det = r_out.detach().to(device=self.r_ema.device, dtype=self.r_ema.dtype)
        self.r_ema.mul_(self.ema_m).add_((1.0 - self.ema_m) * r_det)
    # residual karışım uygulanmış çıkış, girişe göre ne kadar bastırıldı?

    def alpha_etkin(self, x: torch.Tensor, alpha_temiz: torch.Tensor) -> torch.Tensor:
        ratio = (self.r_ema.detach() / max(self.r_min, 1e-12)).clamp(0.0, 1.0)
        ratio = ratio.to(device=x.device, dtype=x.dtype)

        if self.kurtarma_modu == "ratio_floor":
            ratio = ratio.clamp(self.min_kurtarma_orani, 1.0)
            return alpha_temiz * ratio

        alpha_eff = alpha_temiz * ratio
        return alpha_eff.clamp(self.alpha_etkin_min, 1.0)
    # Bu blok, residual karışım katsayısı alpha’yı training sırasında otomatik kısıyor.
    # r_min = “izin verdiğim minimum enerji oranı”

    def forward(self, x: torch.Tensor):
        monitor_stats = None

        if self.return_maps:
            y_ca, ca_map, fusion_w = self.ca(x)
            y = self.coord(y_ca)

            if not self.residual:
                coord_stats = self.coord.last_mask_stats()
                return y, ca_map, fusion_w, coord_stats, None

            alpha_temiz = self.alpha_temiz(x)
            alpha_etkin = alpha_temiz

            if self.training and self.monitor:
                x_std = self.std_batch_ort(x)
                y_std = self.std_batch_ort(y)

                out_tmp = x + alpha_temiz * (y - x)
                out_std = self.std_batch_ort(out_tmp)

                r_block = (y_std / (x_std + 1e-12)).clamp(0.0, 10.0)
                r_out = (out_std / (x_std + 1e-12)).clamp(0.0, 10.0)

                self.r_ema_guncelle(r_out)
                alpha_etkin = self.alpha_etkin(x, alpha_temiz)

                monitor_stats = {
                    "x_std": float(x_std.detach()),
                    "y_std": float(y_std.detach()),
                    "out_std_pre": float(out_std.detach()),
                    "r_block": float(r_block.detach()),
                    "r_out_pre": float(r_out.detach()),
                    "r_ema": float(self.r_ema.detach()),
                    "alpha_temiz": float(alpha_temiz.detach()),
                    "alpha_etkin": float(alpha_etkin.detach()),
                    "kurtarma_modu": self.kurtarma_modu,
                }

            out = x + alpha_etkin * (y - x)
            coord_stats = self.coord.last_mask_stats()
            return out, ca_map, fusion_w, coord_stats, monitor_stats

        y_ca, _ = self.ca(x)
        y = self.coord(y_ca)

        if not self.residual:
            return y

        alpha_temiz = self.alpha_temiz(x)
        alpha_etkin = alpha_temiz

        if self.training and self.monitor:
            x_std = self.std_batch_ort(x)
            out_tmp = x + alpha_temiz * (y - x)
            out_std = self.std_batch_ort(out_tmp)
            r_out = (out_std / (x_std + 1e-12)).clamp(0.0, 10.0)

            self.r_ema_guncelle(r_out)
            alpha_etkin = self.alpha_etkin(x, alpha_temiz)

        out = x + alpha_etkin * (y - x)
        return out

# --- Güvenlik: return_maps açılırsa tuple dönmesin diye ---
def _extract_tensor(out):
    return out[0] if isinstance(out, (tuple, list)) else out


class PreActResidualConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        stride: int = 1,
        norm: str = "bn",        # "bn" / "gn" / "none" (senin make_norm kullanır)
        act: str = "relu",       # "relu" / "silu"
        alpha: float = 1.0,
        skip_norm: bool = False,
    ):
        super().__init__()

        self.norm1 = make_norm(norm, channels)
        if act.lower() == "relu":
            self.act1 = nn.ReLU(inplace=True)
            self.act2 = nn.ReLU(inplace=True)
        elif act.lower() == "silu":
            self.act1 = nn.SiLU(inplace=True)
            self.act2 = nn.SiLU(inplace=True)
        else:
            raise ValueError("act 'relu' veya 'silu' olmalı.")

        self.conv1 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.norm2 = make_norm(norm, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)

        self.proj = None
        if stride != 1:
            layers = [nn.Conv2d(channels, channels, 1, stride=stride, bias=False)]
            if skip_norm:
                layers.append(make_norm(norm, channels))
            self.proj = nn.Sequential(*layers)

        self.register_buffer("alpha_const", torch.tensor(float(alpha)))

    def forward(self, x: torch.Tensor):
        skip = x if self.proj is None else self.proj(x)

        x_main = self.act1(self.norm1(x))
        y = self.conv1(x_main)
        y = self.conv2(self.act2(self.norm2(y)))

        a = self.alpha_const.to(device=x.device, dtype=x.dtype)
        return skip + a * y


class AttnResidualOff_Then_PreAct(nn.Module):
    def __init__(
        self,
        channels: int,
        attn_kwargs: dict | None = None,
        preact_stride: int = 1,
        preact_norm: str = "bn",
        preact_act: str = "relu",
        preact_alpha: float = 1.0,
        preact_skip_norm: bool = False,
    ):
        super().__init__()

        kw = dict(attn_kwargs or {})
        kw["channels"] = channels
        kw["residual"] = False        # <<< residual KAPALI
        kw.setdefault("monitor", False)
        kw.setdefault("return_maps", False) 

        self.attn_fx = CBAMChannelPlusCoord(**kw)

        self.preact = PreActResidualConvBlock(
            channels=channels,
            stride=preact_stride,
            norm=preact_norm,
            act=preact_act,
            alpha=preact_alpha,
            skip_norm=preact_skip_norm,
        )

    def forward(self, x: torch.Tensor):
        y = self.attn_fx(x)
        y = _extract_tensor(y)   # olur da tuple dönerse
        out = self.preact(y)
        return out

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    channels = 64
    attn_kwargs = dict(
        ca_reduction=16,
        ca_fusion="softmax",
        ca_gate="sigmoid",
        ca_temperature=0.9,
        coord_norm="gn",
        coord_dilation=2,
        # return_maps=False, monitor=False zaten default ile kapalı geliyor
    )
    model = AttnResidualOff_Then_PreAct(
        channels=channels,
        attn_kwargs=attn_kwargs,
        preact_norm="bn",
        preact_act="relu",
        preact_alpha=0.7,   
    ).to(device)

    x = torch.randn(8, channels, 32, 32, device=device, requires_grad=True)
    y = model(x)

    print("x:", tuple(x.shape))
    print("y:", tuple(y.shape))
    print("nan?", torch.isnan(y).any().item(), "| inf?", torch.isinf(y).any().item())

    loss = y.mean()
    loss.backward()
    print("loss:", float(loss.item()))
    print("grad preact conv1:", float(model.preact.conv1.weight.grad.norm().item()))
    print("grad preact conv2:", float(model.preact.conv2.weight.grad.norm().item()))
    print("input grad:", float(x.grad.norm().item()))