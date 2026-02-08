import os
import time
import csv
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms
except Exception as e:
    raise RuntimeError("torchvision lazım. pip install torchvision") from e

from torch_cnn.models.reference_net import ReferenceNet


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


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


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_top1(logits.detach(), y) * bs
        n += bs

    return total_loss / max(n, 1), total_acc / max(n, 1)


def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=128)

    # training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)

    # model switches
    p.add_argument("--conv", type=str, default="dynamic", choices=["standard", "dynamic"])
    p.add_argument("--norm", type=str, default="bn", choices=["bn", "gn", "none"])
    p.add_argument("--act", type=str, default="relu", choices=["relu", "silu"])
    p.add_argument("--attn", action="store_true")
    p.add_argument("--no-attn", dest="attn", action="store_false")
    p.set_defaults(attn=True)

    # dynamic params
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--dyn_reduction", type=int, default=4)
    p.add_argument("--dyn_temp", type=float, default=1.0)

    # widths/depths
    p.add_argument("--widths", type=int, nargs=3, default=[64, 128, 256])
    p.add_argument("--depths", type=int, nargs=3, default=[2, 2, 2])

    # outputs
    p.add_argument("--train_csv", type=str, default="./results/train_log.csv")
    p.add_argument("--save_dir", type=str, default="./results/checkpoints")
    p.add_argument("--save_last", action="store_true")

    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ensure_dir(os.path.dirname(args.train_csv))
    ensure_dir(args.save_dir)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"t={elapsed:.1f}s"
        )

        append_row(args.train_csv, dict(
            epoch=epoch,
            conv=args.conv,
            norm=args.norm,
            act=args.act,
            attn=int(args.attn),
            K=args.K,
            dyn_reduction=args.dyn_reduction,
            dyn_temp=args.dyn_temp,
            widths="-".join(map(str, args.widths)),
            depths="-".join(map(str, args.depths)),
            batch_size=args.batch_size,
            lr=args.lr,
            wd=args.wd,
            train_loss=tr_loss,
            train_acc=tr_acc,
        ))

        if args.save_last:
            ckpt_last = os.path.join(args.save_dir, "reference_net_last.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt_last,
            )

    print("done. train_csv:", args.train_csv)


if __name__ == "__main__":
    main()