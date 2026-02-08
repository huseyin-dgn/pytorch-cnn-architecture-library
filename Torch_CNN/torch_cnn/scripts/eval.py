import os
import csv
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from torch_cnn.models.reference_net import ReferenceNet


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def ensure_dir(path: str):
    if path and path.strip():
        os.makedirs(path, exist_ok=True)


def append_row(csv_path: str, row: dict):
    ensure_dir(os.path.dirname(csv_path))
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


@torch.no_grad()
def eval_loop(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_top1(logits, y) * bs
        n += bs

    return total_loss / max(n, 1), total_acc / max(n, 1)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=256)

    p.add_argument("--conv", type=str, default="dynamic", choices=["standard", "dynamic"])
    p.add_argument("--norm", type=str, default="bn", choices=["bn", "gn", "none"])
    p.add_argument("--act", type=str, default="relu", choices=["relu", "silu"])
    p.add_argument("--attn", action="store_true")
    p.add_argument("--no-attn", dest="attn", action="store_false")
    p.set_defaults(attn=True)

    p.add_argument("--K", type=int, default=4)
    p.add_argument("--dyn_reduction", type=int, default=4)
    p.add_argument("--dyn_temp", type=float, default=1.0)

    p.add_argument("--widths", type=int, nargs=3, default=[64, 128, 256])
    p.add_argument("--depths", type=int, nargs=3, default=[2, 2, 2])

    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--strict", action="store_true")

    p.add_argument("--dump", type=str, default="")

    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_ds = datasets.CIFAR10(root=args.data, train=False, download=True, transform=test_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    attn_kwargs = dict(
        ca_reduction=16,
        ca_fusion="softmax",
        ca_gate="sigmoid",
        ca_temperature=0.9,
        coord_norm="gn" if args.norm == "gn" else "bn",
        coord_dilation=2,
        residual=False,
        return_maps=False,
    )

    model = ReferenceNet(
        in_channels=3,
        num_classes=10,
        widths=tuple(args.widths),
        depths=tuple(args.depths),
        conv_kind=args.conv,
        norm=args.norm,
        act=args.act,
        use_attention=bool(args.attn),
        attn_kwargs=attn_kwargs if args.attn else None,
        dynamic_K=args.K,
        dynamic_reduction=args.dyn_reduction,
        dynamic_temperature=args.dyn_temp,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=args.strict)
    print("ckpt:", args.ckpt)
    print("missing keys:", len(missing), "| unexpected keys:", len(unexpected))

    criterion = nn.CrossEntropyLoss()
    loss, acc = eval_loop(model, test_loader, criterion, device)

    print(f"test_loss={loss:.4f} | test_acc={acc:.4f}")

    if args.dump:
        append_row(args.dump, dict(
            ckpt=args.ckpt,
            conv=args.conv,
            norm=args.norm,
            act=args.act,
            attn=int(args.attn),
            K=args.K,
            dyn_reduction=args.dyn_reduction,
            dyn_temp=args.dyn_temp,
            widths="-".join(map(str, args.widths)),
            depths="-".join(map(str, args.depths)),
            test_loss=loss,
            test_acc=acc,
            epoch=ckpt.get("epoch", ""),
        ))
        print("dumped:", args.dump)


if __name__ == "__main__":
    main()
