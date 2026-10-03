import torch
from kmeans import kmeans
from metrics import cluster_acc

data = torch.load("features/cifar10_dino.pt")
x = data["feats"].cuda()

assign, _ = kmeans(x, data["num_classes"], n_init=10)

u = ~data["is_labelled"]
y_true = data["targets"][u].numpy()
y_pred = assign.cpu()[u].numpy()
old_mask = y_true < data["num_old"]

all_acc, old_acc, new_acc = cluster_acc(y_true, y_pred, old_mask)
print(f"All {all_acc*100:.1f}  Old {old_acc*100:.1f}  New {new_acc*100:.1f}")