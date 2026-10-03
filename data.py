import numpy as np
import torchvision
from torchvision import transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# Given image size and crop percentage, return a transform that resizes, crops, converts to tensor, and normalizes the image.
def get_transform(image_size=224, crop_pct=0.875):
    return transforms.Compose([
        transforms.Resize(int(image_size / crop_pct),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

# Given image size and crop percentage, return a transform for training that resizes, randomly crops, flips, converts to tensor, and normalizes the image.
def get_train_transform(image_size=224, crop_pct=0.875):
    return transforms.Compose([
        transforms.Resize(int(image_size / crop_pct),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

# Given a set of targets, the number of old classes, the proportion of labelled data, and a random seed, return a boolean array indicating which samples are labelled.
def make_gcd_split(targets, num_old, prop_labelled=0.5, seed=0):
    rng = np.random.default_rng(seed)
    old_idx = np.where(targets < num_old)[0]
    labelled_idx = rng.choice(old_idx, size=int(prop_labelled * len(old_idx)), replace=False)
    is_labelled = np.zeros(len(targets), dtype=bool)
    is_labelled[labelled_idx] = True
    return is_labelled

# Given a root directory, number of old classes, proportion of labelled data, random seed, and a transform, return the CIFAR-10 dataset, targets, and a boolean array indicating which samples are labelled.
def get_cifar10_gcd(root, num_old=5, prop_labelled=0.5, seed=0, transform=None):
    dataset = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform)
    targets = np.array(dataset.targets)
    is_labelled = make_gcd_split(targets, num_old, prop_labelled, seed)
    return dataset, targets, is_labelled