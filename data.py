import numpy as np
import torchvision
from torchvision import transforms

def get_transform(image_size=224, crop_pct=0.875):
        return transforms.Compose([
                transforms.Resize(int(image_size / crop_pct), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def get_cifar10_gcd(root, num_old=5, prop_labelled=0.5, seed=0):
    dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=get_transform())
    targets = np.array(dataset.targets)
    old_idx = np.where(targets< num_old)[0] 
    rng = np.random.default_rng(seed)
    # Select a subset of the old classes to be labelled
    labelled_idx = rng.choice(old_idx, size= int(prop_labelled * len(old_idx)), replace=False)

    is_labelled = np.zeros(len(dataset), dtype=bool)
    is_labelled[labelled_idx] = True

    return dataset, targets, is_labelled