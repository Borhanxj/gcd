import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ss_kmeans import ss_kmeans
from metrics import cluster_acc, hungarian_acc

np.set_printoptions(linewidth=200)
data = torch.load("features/cifar10_dino.pt")
feats, targets, lab = data["feats"].cuda(), data["targets"].cuda(), data["is_labelled"].cuda()
k, num_old = data["num_classes"], data["num_old"]

assign_u, _ = ss_kmeans(feats[lab], targets[lab], feats[~lab], k=k, num_old=num_old)
pred = assign_u.cpu().numpy()
y_true = data["targets"][~data["is_labelled"]].numpy()
old = y_true < num_old

# (A) Old accuracy WITHOUT Hungarian: cluster j < num_old *is* Old class j by construction
print("Direct Old acc (no matching):", (pred[old] == y_true[old]).mean())

# (B) Paper-style: one matching over everything
print("One matching  All/Old/New:", cluster_acc(y_true, pred, old))

# (C) Separate matching for Old and New
print("Separate matching  Old:", hungarian_acc(y_true[old], pred[old]),
      " New:", hungarian_acc(y_true[~old], pred[~old]))

# (D) Where do images go? rows = clusters, cols = true classes
w = np.zeros((k, k), dtype=int)
np.add.at(w, (pred, y_true), 1)
print(w)
row, col = linear_sum_assignment(w, maximize=True)
print("cluster -> class:", dict(zip(row.tolist(), col.tolist())))