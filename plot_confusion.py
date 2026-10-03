import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment
from kmeans import kmeans
from ss_kmeans import ss_kmeans

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

def confusion(pred, y_true, k):
    w = np.zeros((k, k), dtype=int)          # rows = clusters, cols = true classes
    np.add.at(w, (pred, y_true), 1)
    return w

def report(w, num_old, anchored, name):
    row, col = linear_sum_assignment(w, maximize=True)
    match = dict(zip(row.tolist(), col.tolist()))
    print(f"\n=== {name} ===")
    for c in range(w.shape[0]):
        tag = f"anchor: {CLASSES[c]}" if anchored and c < num_old else "free"
        top = np.argsort(w[c])[::-1][:3]
        content = ", ".join(f"{CLASSES[t]} {w[c, t]}" for t in top if w[c, t] > 0)
        print(f"C{c} [{tag:>18}]  size {w[c].sum():5d} | {content:45s} | matched -> {CLASSES[match[c]]}")
    return match

def plot(ax, w, match, num_old, anchored, title):
    k = w.shape[0]
    frac = w / w.sum(axis=0, keepdims=True)  # each column sums to 1: where did this class go?
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)

    for i in range(k):
        for j in range(k):
            if w[i, j] > 0:
                ax.text(j, i, w[i, j], ha="center", va="center", fontsize=7,
                        color="white" if frac[i, j] > 0.5 else "black")

    for c, t in match.items():               # red box = Hungarian matching
        ax.add_patch(Rectangle((t - 0.5, c - 0.5), 1, 1,
                               fill=False, edgecolor="red", lw=2))

    ax.set_xticks(range(k))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    for i, lbl in enumerate(ax.get_xticklabels()):
        lbl.set_color("tab:blue" if i < num_old else "tab:orange")

    ax.set_yticks(range(k))
    ax.set_yticklabels([f"C{c} ({CLASSES[c]} anchor)" if anchored and c < num_old else f"C{c}"
                        for c in range(k)], fontsize=8)
    ax.set_xlabel("true class  (blue = Old, orange = New)")
    ax.set_ylabel("cluster")
    ax.set_title(title)

def main():
    data = torch.load("features/cifar10_dino.pt")
    feats, targets, lab = data["feats"].cuda(), data["targets"].cuda(), data["is_labelled"].cuda()
    k, num_old = data["num_classes"], data["num_old"]

    plain_all, _ = kmeans(feats, k)
    plain_u = plain_all[~lab].cpu().numpy()
    ss_u, _ = ss_kmeans(feats[lab], targets[lab], feats[~lab], k=k, num_old=num_old)
    ss_u = ss_u.cpu().numpy()

    y_true = data["targets"][~data["is_labelled"]].numpy()   # evaluation only

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, pred, anchored, name in [(axes[0], plain_u, False, "Plain k-means"),
                                     (axes[1], ss_u, True, "Semi-supervised k-means")]:
        w = confusion(pred, y_true, k)
        match = report(w, num_old, anchored, name)
        plot(ax, w, match, num_old, anchored, name)

    plt.tight_layout()
    plt.savefig("confusion_cifar10.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()