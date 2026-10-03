import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

# Class for generating two views of the same image (for contrastive learning)
class TwoViewTransform:
    def __init__(self, base_transform):
        self.base = base_transform

    def __call__(self, img):
        return [self.base(img), self.base(img)]

# Class for the GCD training dataset, which returns two views of each image, its label (or -1 if unlabelled), and a boolean indicating whether it is labelled
# Even if it is unlabelled, we still have contrastive learning views for it, so we can use it in the contrastive loss.
# So during training, we can use the labelled samples for supervised loss and all samples for contrastive loss.
class GCDTrainDataset(Dataset):
    def __init__(self, base_dataset, targets, is_labelled, transform):
        self.base = base_dataset          # returns PIL images (transform=None)
        self.targets = targets
        self.is_labelled = is_labelled
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img, _ = self.base[i]
        views = self.transform(img)
        label = int(self.targets[i]) if self.is_labelled[i] else -1
        return views, label, bool(self.is_labelled[i])

# This function generates a sampler that samples labelled and unlabelled images with equal probability, so that each batch has a balanced number of labelled and unlabelled images.
def make_sampler(is_labelled):
    n_lab, n_unlab = is_labelled.sum(), (~is_labelled).sum()
    weights = np.where(is_labelled, 1.0, n_lab / n_unlab)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(is_labelled), replacement=True)
                                 