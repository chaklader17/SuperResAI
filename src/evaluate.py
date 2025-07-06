import os
import torch
import cv2
from torchvision.utils import save_image
from models.baseline_cnn import SRCNN  # or change to your actual model file
from dataset_loader import load_datasets

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRCNN().to(device)
model.load_state_dict(torch.load("results/srcnn_model.pth", map_location=device, weights_only=True))
model.eval()

# Load test dataset
_, _, test_set = load_datasets("data/low_res", "data/high_res")

# Create output dir
os.makedirs("results/comparisons", exist_ok=True)

# Evaluate and save comparisons
with torch.no_grad():
    for idx in range(len(test_set)):
        low_res, high_res = test_set[idx]

        input_tensor = low_res.unsqueeze(0).to(device)  # Add batch dimension
        output = model(input_tensor).squeeze(0).cpu()   # Remove batch dimension

        # Convert tensors to images and save
        save_image(low_res.clamp(0, 1), f"results/comparisons/test_{idx}_low.png")
        save_image(output.clamp(0, 1), f"results/comparisons/test_{idx}_restored.png")
        save_image(high_res.clamp(0, 1), f"results/comparisons/test_{idx}_high.png")

print("✅ Evaluation complete. Comparisons saved in 'results/comparisons/'")
print(low_res.min(), low_res.max())
print(output.min(), output.max())
