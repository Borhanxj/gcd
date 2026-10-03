import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import get_cifar10_gcd

@torch.no_grad()
def extract(model, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats = []
    for images, _ in loader:
        feats.append(model(images.to(device)).cpu())
    return torch.cat(feats)

def main():
    device = "cuda"
    model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").to(device).eval()

    dataset, targets, is_labelled = get_cifar10_gcd("./data")
    feats = F.normalize(extract(model, dataset, device), dim=-1)

    print("features:", feats.shape)
    print("|D_L| =", is_labelled.sum(), " |D_U| =", (~is_labelled).sum())
    print("D_U old:", ((targets < 5) & ~is_labelled).sum(),
          " D_U new:", ((targets >= 5) & ~is_labelled).sum())

    os.makedirs("features", exist_ok=True)
    torch.save({"feats": feats,
                "targets": torch.from_numpy(targets),
                "is_labelled": torch.from_numpy(is_labelled),
                "num_old": 5, "num_classes": 10},
               "features/cifar10_dino.pt")

if __name__ == "__main__":
    main()