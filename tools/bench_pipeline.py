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
    p.add_argument("--pin-memory", type=int, default=0,
                   help="DataLoader(pin_memory=...). Only useful together with "
                        "--non-blocking 1, and only if the tensor actually transferred "
                        "is the pinned one (see [E2]).")
    p.add_argument("--non-blocking", type=int, default=0,
                   help="use .to(..., non_blocking=True) for the host-to-device copy")
    p.add_argument("--threads", type=int, default=0,
                   help="torch.set_num_threads(N); 0 leaves the default. Use 1 to test "
                        "intra-op thread oversubscription on 3x32x32 tensors.")
    args = p.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

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
                             persistent_workers=args.workers > 0,
                             pin_memory=bool(args.pin_memory))

    print("=== [C] passes over the train loader, no model ===")
    # Two passes: with persistent_workers=True the first one pays the spawning of the
    # worker processes, which the trainer pays once per run and not once per epoch.
    # Compare [E] against C2, never against C1.
    c = None
    for k in (1, 2):
        t = time.perf_counter()
        nb = 0
        for _ in loader:
            nb += 1
        ck = time.perf_counter() - t
        tag = "C1 (includes worker spawn)" if k == 1 else "C2 (steady state)"
        print(f"  {tag:<28} {ck:.2f} s for {nb} batches  ({ck / nb * 1e3:.2f} ms/batch)")
        c = ck

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

    print(f"  (pin_memory={bool(args.pin_memory)}, "
          f"non_blocking={bool(args.non_blocking)})")
    nbk = bool(args.non_blocking)

    print("=== [E] loader + transfer + step, as the trainer runs it ===")
    # loader_pre_hooks.classification does x.to(device=..., dtype=...): when the loader
    # yields float32 and dtype is float64, the cast and the transfer are requested in one
    # call, and whether the pinned buffer is the thing actually DMA'd is not guaranteed.
    # [E2] below separates the two so the question is answered by measurement.
    t = time.perf_counter()
    nb = 0
    for xb, yb in loader:
        xb = xb.to(dev, dtype, non_blocking=nbk)
        yb = yb.to(dev, non_blocking=nbk)
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(xb), yb)
        loss.backward()
        opt.step()
        loss.item()
        nb += 1
    e = time.perf_counter() - t
    print(f"  {e:.2f} s for {nb} batches  ({e / nb * 1e3:.2f} ms/batch)")

    print("=== [E2] same, but transfer first and cast on the device ===")
    t = time.perf_counter()
    nb = 0
    for xb, yb in loader:
        xb = xb.to(dev, non_blocking=nbk).to(dtype)   # DMA the pinned tensor, then cast
        yb = yb.to(dev, non_blocking=nbk)
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(xb), yb)
        loss.backward()
        opt.step()
        loss.item()
        nb += 1
    e2 = time.perf_counter() - t
    print(f"  {e2:.2f} s for {nb} batches  ({e2 / nb * 1e3:.2f} ms/batch, "
          f"{(e - e2) / e * 100:+.0f} % vs [E])")

    print("=== [D2] same step WITHOUT the per-step .item() ===")
    # training_hydra.py calls .item() on every iteration (lines 461-463, 520-524), which
    # forces a host-device sync 450 times per epoch and prevents any overlap between the
    # launch of step n+1 and the execution of step n.
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        lossf(model(x), y).backward()
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    acc = torch.zeros((), device=dev, dtype=dtype)
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()
        acc += loss.detach()      # accumulate on the device, read once at the end
    acc.item()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    d2 = (time.perf_counter() - t) / args.steps
    delta = (d - d2) * 450 * 1e3
    verdict = (f"{(d - d2) / d * 100:.0f} % faster than [D], {delta:.0f} ms saved/epoch"
               if d2 < d else
               f"SLOWER than [D] by {(d2 - d) / d * 100:.0f} % -- implausible, "
               f"this run is contaminated (warm-up or a shared node); discard it")
    print(f"  {d2 * 1e3:.2f} ms/step  ->  {d2 * 450 * 1e3:.0f} ms for 450 steps "
          f"({verdict})")

    print("=== [F] epoch with the dataset resident on the device, no DataLoader ===")
    # Legitimate only because data_augm=False makes transform_train deterministic
    # (ToTensor + Normalize): precomputing it once is exactly equivalent, up to the
    # shuffling order. Memory: 50000*3*32*32*4 B = 614 MB in fp32, 1.2 GB in fp64.
    t = time.perf_counter()
    big = torch.stack([trainset[i][0] for i in range(len(trainset))])
    labels = torch.tensor(trainset.targets)
    prep = time.perf_counter() - t
    try:
        gx = big.to(dev, dtype)
        gy = labels.to(dev)
        print(f"  one-off preparation: {prep:.1f} s, {gx.numel() * gx.element_size() / 2**20:.0f} MiB on device")
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        perm = torch.randperm(gx.shape[0], device=dev)
        nb = 0
        for i in range(0, gx.shape[0], args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(gx[idx]), gy[idx])
            loss.backward()
            opt.step()
            acc += loss.detach()
            nb += 1
        acc.item()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        f = time.perf_counter() - t
        print(f"  {f:.2f} s for {nb} batches  ({f / nb * 1e3:.2f} ms/batch)")
    except RuntimeError as exc:
        print(f"  skipped: {exc}")

    print("=== reading ===")
    print(f"  host-side share of [E]: {(e - d * nb) / e * 100:.0f} %  "
          f"(device work = {d * nb:.2f} s of {e:.2f} s)")
    print(f"  an epoch also runs valid+test (150 more batches): "
          f"add roughly {c / nb * 150:.2f} s of loading")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
