import argparse, json, os, time, sys
print("[DEBUG 1/10] Importing Python modules...", flush=True)

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset

print("[DEBUG 2/10] Importing custom project modules...", flush=True)
from data import get_cifar10_gcd, get_transform, get_train_transform
from train_data import TwoViewTransform, GCDTrainDataset, make_sampler
from model import GCDModel
from losses import gcd_loss
from kmeans import kmeans
from metrics import cluster_acc, hungarian_acc

print("[DEBUG 3/10] Imports successfully loaded.", flush=True)

NUM_OLD, NUM_CLASSES = 5, 10

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./data")
    p.add_argument("--out", default="runs/cifar10")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-5)
    p.add_argument("--lam", type=float, default=0.35)
    p.add_argument("--t_u", type=float, default=1.0)
    p.add_argument("--t_s", type=float, default=0.07)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--du_eval_every", type=int, default=10)   # 0 = off
    p.add_argument("--max_steps", type=int, default=0)        # >0: short debug epochs
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

@torch.no_grad()
def extract(model, loader, tag="loader"):
    print(f"  [DEBUG extract] Starting feature extraction for '{tag}' ({len(loader)} total batches)...", flush=True)
    model.eval()
    feats, ys = [], []
    for step, (x, y) in enumerate(loader):
        if step % 10 == 0 or step == len(loader) - 1:
            print(f"  [DEBUG extract] '{tag}' -> Batch {step + 1}/{len(loader)}", flush=True)
        feats.append(F.normalize(model.features(x.cuda()), dim=-1).cpu())
        ys.append(y)
    model.train()
    print(f"  [DEBUG extract] Finished batch iteration for '{tag}'. Concatenating features...", flush=True)
    return torch.cat(feats), torch.cat(ys).numpy()

def evaluate(model, val_loader, du_loader):
    out = {}
    print(" [DEBUG evaluate] Starting evaluation on val_loader...", flush=True)
    f, y = extract(model, val_loader, tag="val_loader")
    print(" [DEBUG evaluate] Moving val features to GPU and calling kmeans()...", flush=True)
    pred, _ = kmeans(f.cuda(), NUM_OLD)
    print(" [DEBUG evaluate] Computing hungarian_acc...", flush=True)
    out["val_acc"] = hungarian_acc(y, pred.cpu().numpy())
    print(" [DEBUG evaluate] val_loader evaluation complete.", flush=True)
    
    if du_loader is not None:
        print(" [DEBUG evaluate] Starting evaluation on du_loader...", flush=True)
        f, y = extract(model, du_loader, tag="du_loader")
        print(" [DEBUG evaluate] Moving du features to GPU and calling kmeans()...", flush=True)
        pred, _ = kmeans(f.cuda(), NUM_CLASSES)
        print(" [DEBUG evaluate] Computing cluster_acc...", flush=True)
        a, o, n = cluster_acc(y, pred.cpu().numpy(), y < NUM_OLD)
        out.update(du_all=a, du_old=o, du_new=n)
        print(" [DEBUG evaluate] du_loader evaluation complete.", flush=True)
    return out

def train_one_epoch(model, loader, opt, args):
    model.train()
    sums, n = np.zeros(3), 0
    print(f" [DEBUG train_epoch] Beginning batch loop...", flush=True)
    for step, (views, labels, is_lab) in enumerate(loader):
        if args.max_steps and step >= args.max_steps:
            print(f" [DEBUG train_epoch] Hit max_steps ({args.max_steps}), breaking step loop.", flush=True)
            break
        
        if step == 0 or (step + 1) % 10 == 0:
            print(f" [DEBUG train_epoch] Step {step + 1}/{len(loader)}", flush=True)

        x = torch.cat(views).cuda(non_blocking=True)
        labels, is_lab = labels.cuda(), is_lab.cuda()

        _, proj = model(x)
        loss, l_u, l_s = gcd_loss(proj, labels, is_lab, args.lam, args.t_u, args.t_s)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        sums += [loss.item(), l_u.item(), l_s.item()]
        n += 1
    return sums / n

def main():
    print("[DEBUG 4/10] Entering main(). Parsing arguments...", flush=True)
    args = get_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print("[DEBUG 5/10] Checking PyTorch GPU/ROCm Status...", flush=True)
    print(f"  PyTorch CUDA Available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"  Device Name: {torch.cuda.get_device_name(0)}", flush=True)

    print("[DEBUG 6/10] Loading CIFAR-10 training datasets...", flush=True)
    base, targets, is_lab = get_cifar10_gcd(args.root, seed=args.seed, transform=None)
    train_ds = GCDTrainDataset(base, targets, is_lab, TwoViewTransform(get_train_transform()))
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        sampler=make_sampler(is_lab),
        num_workers=args.num_workers, 
        drop_last=True, 
        pin_memory=(args.num_workers > 0)
    )

    print("[DEBUG 7/10] Loading CIFAR-10 validation & du datasets...", flush=True)
    test = torchvision.datasets.CIFAR10(args.root, train=False, download=True, transform=get_transform())
    val_idx = np.where(np.array(test.targets) < NUM_OLD)[0]
    val_loader = DataLoader(Subset(test, val_idx), batch_size=256, num_workers=args.num_workers)

    du_loader = None
    if args.du_eval_every:
        train_eval = torchvision.datasets.CIFAR10(args.root, train=True, transform=get_transform())
        du_loader = DataLoader(Subset(train_eval, np.where(~is_lab)[0]), batch_size=256, num_workers=args.num_workers)

    print("[DEBUG 8/10] Instantiating GCDModel & pushing to GPU (ROCm HIP)...", flush=True)
    t_model_start = time.time()
    model = GCDModel().cuda()
    print(f"[DEBUG 8/10] Model moved to GPU in {time.time() - t_model_start:.2f} seconds.", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 1e-3)

    log_path = os.path.join(args.out, "log.jsonl")

    print("[DEBUG 9/10] Triggering Baseline Evaluation (Epoch 0)...", flush=True)
    start = {"epoch": 0, **evaluate(model, val_loader, du_loader)}
    print(f"[DEBUG 9/10] Baseline Eval Results: {start}", flush=True)
    
    with open(log_path, "w") as f:
        f.write(json.dumps(start) + "\n")

    print("[DEBUG 10/10] Entering Training Loop...", flush=True)
    best = -1
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---", flush=True)
        t0 = time.time()
        loss, l_u, l_s = train_one_epoch(model, train_loader, opt, args)
        sched.step()
        log = {
            "epoch": epoch, "loss": loss, "l_u": l_u, "l_s": l_s,
            "lr": sched.get_last_lr()[0], "time": time.time() - t0
        }

        if epoch % args.eval_every == 0:
            use_du = args.du_eval_every and epoch % args.du_eval_every == 0
            log.update(evaluate(model, val_loader, du_loader if use_du else None))
            if log["val_acc"] > best:
                best = log["val_acc"]
                torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))

        print(f"Epoch {epoch} Complete -> Log: {log}", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(log) + "\n")

    torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
    print("ALL DONE SUCCESSFUL!", flush=True)

if __name__ == "__main__":
    main()