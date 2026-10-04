import argparse, json, os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import get_cifar10_gcd, get_transform
from model import GCDModel
from kmeans import kmeans
from ss_kmeans import ss_kmeans
from metrics import cluster_acc, hungarian_acc

NUM_OLD, NUM_CLASSES = 5, 10

@torch.no_grad()
def extract(model, loader):
    model.eval()
    return torch.cat([F.normalize(model.features(x.cuda()), dim=-1).cpu() for x, _ in loader])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="")              # empty = raw DINO
    p.add_argument("--root", default="./data")
    p.add_argument("--seed", type=int, default=0)     # MUST match the training seed
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_feats", default="")
    args = p.parse_args()

    model = GCDModel().cuda()
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))

    dataset, targets, is_lab = get_cifar10_gcd(args.root, seed=args.seed, transform=get_transform())
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=args.num_workers)
    feats = extract(model, loader).cuda()
    t = torch.from_numpy(targets).cuda()
    lab = torch.from_numpy(is_lab).cuda()

    # clustering uses D_L labels only
    plain, _ = kmeans(feats, NUM_CLASSES)
    ss_u, _ = ss_kmeans(feats[lab], t[lab], feats[~lab], k=NUM_CLASSES, num_old=NUM_OLD)
    preds = {"kmeans": plain[~lab].cpu().numpy(), "ss_kmeans": ss_u.cpu().numpy()}

    # evaluation uses D_U labels
    y_true = targets[~is_lab]
    old = y_true < NUM_OLD
    results = {}
    for name, pred in preds.items():
        a, o, n = cluster_acc(y_true, pred, old)
        results[name] = {"all": a, "old": o, "new": n,
                         "old_sep": hungarian_acc(y_true[old], pred[old]),
                         "new_sep": hungarian_acc(y_true[~old], pred[~old])}
        r = results[name]
        print(f"{name:10s} All {a*100:.1f}  Old {o*100:.1f}  New {n*100:.1f}   "
              f"| separate matching: Old {r['old_sep']*100:.1f}  New {r['new_sep']*100:.1f}")

    if args.ckpt:
        with open(os.path.join(os.path.dirname(args.ckpt), "eval.json"), "w") as f:
            json.dump(results, f, indent=2)
    if args.save_feats:
        torch.save({"feats": feats.cpu(), "targets": torch.from_numpy(targets),
                    "is_labelled": torch.from_numpy(is_lab),
                    "num_old": NUM_OLD, "num_classes": NUM_CLASSES}, args.save_feats)

if __name__ == "__main__":
    main()