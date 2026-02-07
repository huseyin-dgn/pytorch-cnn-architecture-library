# train_ddp_syncbn.py
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from model_syncbn import SyncBNNet

def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def main():
    local_rank = setup()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    ds = datasets.FakeData(size=5000, image_size=(3, 224, 224), num_classes=10, transform=tfm)
    sampler = DistributedSampler(ds, shuffle=True)
    dl = DataLoader(ds, batch_size=8, sampler=sampler, num_workers=2, pin_memory=True)

    model = SyncBNNet(num_classes=10, width=32).cuda(local_rank)
    
    #  SyncBN dönüşümü (DDP'den ÖNCE)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    #  DDP wrap
    model = DDP(model, device_ids=[local_rank])

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):
        sampler.set_epoch(epoch)
        for x, y in dl:
            x = x.cuda(local_rank, non_blocking=True)
            y = y.cuda(local_rank, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

        if local_rank == 0:
            print(f"epoch {epoch} done")

    cleanup()

if __name__ == "__main__":
    main()