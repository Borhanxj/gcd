import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from data import get_cifar10_gcd, get_train_transform, MEAN, STD
from train_data import TwoViewTransform, GCDTrainDataset, make_sampler

def main():
    base, targets, is_labelled = get_cifar10_gcd("./data", transform=None)
    ds = GCDTrainDataset(base, targets, is_labelled, TwoViewTransform(get_train_transform()))
    loader = DataLoader(ds, batch_size=128, sampler=make_sampler(is_labelled),
                        num_workers=4, drop_last=True)

    views, labels, mask = next(iter(loader))
    print("view shapes:", views[0].shape, views[1].shape)
    print("labelled fraction in batch:", mask.float().mean().item())
    print("labels of unlabelled images:", labels[~mask].unique().tolist())
    print("labelled classes in batch:", labels[mask].unique().tolist())

    mean, std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
    pairs = torch.stack([views[0][:8], views[1][:8]], dim=1).flatten(0, 1)
    save_image(pairs * std + mean, "view_pairs.png", nrow=8)

if __name__ == "__main__":
    main()