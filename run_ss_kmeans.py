import torch
from ss_kmeans import ss_kmeans
from metrics import cluster_acc

data = torch.load("features/cifar10_dino.pt")
feats = data["feats"].cuda()
targets = data["targets"].cuda()
lab = data["is_labelled"].cuda()

x_l, y_l = feats[lab], targets[lab]
x_u = feats[~lab]

assign_u, _ = ss_kmeans(x_l, y_l, x_u, k=data["num_classes"], num_old=data["num_old"])

y_true = data["targets"][~data["is_labelled"]].numpy()
old_mask = y_true < data["num_old"]
all_acc, old_acc, new_acc = cluster_acc(y_true, assign_u.cpu().numpy(), old_mask)
print(f"All {all_acc*100:.1f}  Old {old_acc*100:.1f}  New {new_acc*100:.1f}")