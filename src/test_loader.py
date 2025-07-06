from dataset_loader import load_datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Load datasets
train_set, val_set, test_set = load_datasets("data/low_res", "data/high_res")

print(f"Train size: {len(train_set)}")
print(f"Val size:   {len(val_set)}")
print(f"Test size:  {len(test_set)}")

# View a sample
low, high = train_set[0]

plt.subplot(1, 2, 1)
plt.imshow(low.permute(1, 2, 0))
plt.title("Low-Res")

plt.subplot(1, 2, 2)
plt.imshow(high.permute(1, 2, 0))
plt.title("High-Res")

plt.show()
