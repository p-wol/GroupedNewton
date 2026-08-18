#!/usr/bin/env python
"""tools/bench_pipeline.py -- decompose one LeNet/CIFAR-10 epoch into host and device time.

Purpose: LeNet-5 on CIFAR-10 at batch 100 is roughly 2e8 MACs per training step. Even at
a tenth of a V100's fp32 peak, that is under 1 ms per step, i.e. well under a second for
the 450 steps of an epoch. An epoch that takes 5 s or 16 s is therefore dominated by
host-side work, and comparing two machines on the *total* tells you nothing about which
part differs. This script measures the parts separately.

Run the same command on both machines, with the same --root data and --dtype:

    python tools/bench_pipeline.py --root $DSDIR --workers 0
    python tools/bench_pipeline.py --root $DSDIR --workers 9

It reports, in order:
  [A] dataset construction (torchvision reads the pickles into RAM)
  [B] per-sample transform cost in the main process (ToTensor + Normalize)
  [C] one full pass over the train loader, no model at all
  [D] forward+backward+step on a batch already resident on the device, synchronized
  [E] the loop as the trainer runs it: loader + transfer + step

and the implied epoch time. Nothing is written to disk; no data is downloaded
(download=False, as in src/grnewt/datasets.py).
"""

from __future__ import annotations

import argparse
import os
import platform
import time

import torch
import torchvision
from torch.utils import data
from torchvision import transforms

# Same shapes as grnewt.models.LeNet with args '6-16-120-84-10' on 3x32x32 inputs.
class LeNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(5 * 5 * 16, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = torch.nn.functional.max_pool2d(torch.tanh(self.conv1(x)), 2)
        x = torch.nn.functional.max_pool2d(torch.tanh(self.conv2(x)), 2)
        x = x.flatten(1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="CIFAR-10 root (<root>/cifar-10-batches-py)")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--dtype", type=int, default=32, choices=(32, 64))
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--deterministic", type=int, default=1,
                   help="1 reproduces set_seeds(): cudnn.deterministic=True, benchmark=False")
    args = p.parse_args()

    torch.backends.cudnn.deterministic = bool(args.deterministic)
    torch.backends.cudnn.benchmark = not bool(args.deterministic)
    dtype = torch.float32 if args.dtype == 32 else torch.float64
    dev = torch.device(args.device)

    print("=== environment ===")
    print(f"  host                = {platform.node()}")
    print(f"  cpu                 = {platform.processor() or 'unknown'}")
    print(f"  os.cpu_count        = {os.cpu_count()}")
    print(f"  torch.get_num_threads = {torch.get_num_threads()}")
    print(f"  OMP_NUM_THREADS     = {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  torch               = {torch.__version__}  (cuda {torch.version.cuda})")
    if dev.type == "cuda":
        print(f"  gpu                 = {torch.cuda.get_device_name(0)}")
        print(f"  arch_list           = {torch.cuda.get_arch_list()}")
    print(f"  dtype               = {dtype}, batch_size = {args.batch_size}, "
          f"workers = {args.workers}, cudnn.deterministic = {args.deterministic}")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    print("=== [A] dataset construction ===")
    t = time.perf_counter()
    trainset = torchvision.datasets.CIFAR10(root=args.root, train=True, download=False,
                                            transform=tf)
    a = time.perf_counter() - t
    print(f"  {a:.2f} s for {len(trainset)} samples")

    print("=== [B] per-sample transform, main process ===")
    n = 5000
    t = time.perf_counter()
    for i in range(n):
        trainset[i]
    b = (time.perf_counter() - t) / n
    print(f"  {b * 1e6:.1f} us/sample  ->  {b * 60000 * 1e3:.0f} ms for the 60000 "
          f"sample-transforms of one epoch (train+valid+test)")

    loader = data.DataLoader(trainset, args.batch_size, shuffle=True,
                             num_workers=args.workers,
                             persistent_workers=args.workers > 0)

    print("=== [C] one pass over the train loader, no model ===")
    t = time.perf_counter()
    nb = 0
    for _ in loader:
        nb += 1
    c = time.perf_counter() - t
    print(f"  {c:.2f} s for {nb} batches  ({c / nb * 1e3:.2f} ms/batch)")

    model = LeNet().to(dev, dtype)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()

    print("=== [D] step on a batch already on the device ===")
    x = torch.randn(args.batch_size, 3, 32, 32, device=dev, dtype=dtype)
    y = torch.randint(0, 10, (args.batch_size,), device=dev)
    for _ in range(20):  # warm-up: cudnn algorithm selection, allocator, autotune
        opt.zero_grad(set_to_none=True)
        lossf(model(x), y).backward()
        opt.step()
    torch.cuda.synchronize() if dev.type == "cuda" else None
    t = time.perf_counter()
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()
        loss.item()          # the trainer calls .item() every step: this forces a sync
    torch.cuda.synchronize() if dev.type == "cuda" else None
    d = (time.perf_counter() - t) / args.steps
    print(f"  {d * 1e3:.2f} ms/step (including the .item() sync)  ->  "
          f"{d * 450 * 1e3:.0f} ms for 450 steps")

    print("=== [E] loader + transfer + step, as the trainer runs it ===")
    t = time.perf_counter()
    nb = 0
    for xb, yb in loader:
        xb = xb.to(dev, dtype)
        yb = yb.to(dev)
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(xb), yb)
        loss.backward()
        opt.step()
        loss.item()
        nb += 1
    e = time.perf_counter() - t
    print(f"  {e:.2f} s for {nb} batches  ({e / nb * 1e3:.2f} ms/batch)")

    print("=== reading ===")
    print(f"  host-side share of [E]: {(e - d * nb) / e * 100:.0f} %  "
          f"(device work = {d * nb:.2f} s of {e:.2f} s)")
    print(f"  an epoch also runs valid+test (150 more batches): "
          f"add roughly {c / nb * 150:.2f} s of loading")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
