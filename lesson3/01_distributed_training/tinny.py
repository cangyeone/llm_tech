
"""
export NCCL_SOCKET_IFNAME=eth0   # 换成你的网卡名；单网卡可省略
torchrun --standalone --nproc_per_node=8 tinny.py
"""
# ddp_minimal.py
import os, torch, torch.distributed as dist
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler

class ToySet(Dataset):
    def __len__(self): return 10000
    def __getitem__(self, i):
        x = torch.randn(32)           # 随机特征
        y = (x.sum() > 0).long()      # 二分类标签
        return x, y

def setup():
    dist.init_process_group(backend="nccl")

def cleanup():
    dist.destroy_process_group()

def main():
    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 模型/损失/优化器
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    crit = nn.CrossEntropyLoss().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    # 数据 + 采样器
    ds = ToySet()
    sampler = DistributedSampler(ds, shuffle=True)
    dl = DataLoader(ds, batch_size=64, sampler=sampler, num_workers=2, pin_memory=True)

    model.train()
    for epoch in range(2):
        sampler.set_epoch(epoch)
        for step, (x, y) in enumerate(dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            if step % 100 == 0 and dist.get_rank() == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), "ddp_minimal.pt")
        print("✅ saved: ddp_minimal.pt")
    cleanup()

if __name__ == "__main__":
    main()
