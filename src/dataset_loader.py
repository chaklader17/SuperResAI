# src/dataset_loader.py

import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Dataset class
class SuperResDataset(Dataset):
    def __init__(self, low_paths, high_paths, transform=None):
        self.low_paths = low_paths
        self.high_paths = high_paths
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low_img = Image.open(self.low_paths[idx]).convert("RGB")
        high_img = Image.open(self.high_paths[idx]).convert("RGB")
        return self.transform(low_img), self.transform(high_img)

# Split and load datasets
def load_datasets(low_res_dir, high_res_dir, split=(0.6, 0.2, 0.2), seed=42):
    # Step 1: Get all matching filenames
    filenames = sorted([
        f for f in os.listdir(low_res_dir)
        if os.path.exists(os.path.join(high_res_dir, f))
    ])

    # Step 2: Shuffle filenames
    random.seed(seed)
    random.shuffle(filenames)

    # Step 3: Split
    total = len(filenames)
    train_end = int(split[0] * total)
    val_end = train_end + int(split[1] * total)

    train_files = filenames[:train_end]
    val_files = filenames[train_end:val_end]
    test_files = filenames[val_end:]

    # Step 4: Convert to full paths
    def make_paths(file_list):
        low = [os.path.join(low_res_dir, f) for f in file_list]
        high = [os.path.join(high_res_dir, f) for f in file_list]
        return low, high

    train = make_paths(train_files)
    val = make_paths(val_files)
    test = make_paths(test_files)

    # Step 5: Return datasets
    return (
        SuperResDataset(*train),
        SuperResDataset(*val),
        SuperResDataset(*test),
    )