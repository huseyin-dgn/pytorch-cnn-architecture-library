# PyTorch CNN Architecture Library

Modüler CNN blokları (DynamicConv, Attention, Residual, Norm) ile deney yapmaya uygun referans yapı.

---

## Kurulum

```bash
pip install torch torchvision
```

Repo kök dizininde çalıştığından emin ol:

```bash
cd pytorch-cnn-architecture-library
```

Import testi:

```bash
python -c "import torch_cnn; print('ok')"
```

Eğer import hatası alırsan:

```bash
set PYTHONPATH=.
```

---

## Train (Sadece Eğitim)

Bu komut modeli eğitir ve her epoch sonunda checkpoint günceller.

**Çıktılar**

- Eğitim logu → `results/train_log.csv`
- Checkpoint → `results/checkpoints/reference_net_last.pt`

```bash
python scripts/train.py --epochs 10 --conv dynamic --norm bn --act relu --attn
```

GroupNorm + SiLU:

```bash
python scripts/train.py --epochs 10 --conv dynamic --norm gn --act silu --attn
```

Attention kapalı:

```bash
python scripts/train.py --epochs 10 --conv dynamic --no-attn
```

---

## Eval (Checkpoint Ölçüm)

Checkpoint’i yükler ve CIFAR-10 test setinde ölçüm yapar.

```bash
python scripts/eval.py --ckpt results/checkpoints/reference_net_last.pt
```

Sonucu CSV’ye yazdırmak:

```bash
python scripts/eval.py --ckpt results/checkpoints/reference_net_last.pt --dump results/eval_rows.csv
```

**Not:** Eval sırasında train’de kullandığın model argümanları ile aynı konfigürasyonu vermelisin (`--conv --norm --act --attn --K ...`).
