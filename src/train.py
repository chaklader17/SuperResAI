import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from models.baseline_cnn import SRCNN
from dataset_loader import load_datasets
from metrics import calculate_psnr, calculate_ssim

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
num_epochs = 40
batch_size = 4
learning_rate = 1e-4

# Logging dictionary
history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "psnr": [],
    "ssim": []
}

# Load datasets
train_set, val_set, test_set = load_datasets("data/low_res", "data/high_res")
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# Initialize model, loss, optimizer
model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for low_res, high_res in train_loader:
        low_res = low_res.to(device)
        high_res = high_res.to(device)

        optimizer.zero_grad()
        output = model(low_res)
        loss = criterion(output, high_res)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    total_psnr = 0
    total_ssim = 0

    with torch.no_grad():
        for low_res, high_res in val_loader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            output = model(low_res)
            loss = criterion(output, high_res)
            val_loss += loss.item()
            total_psnr += calculate_psnr(output, high_res)
            total_ssim += calculate_ssim(output, high_res)

    val_loss /= len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)

    # Log results
    history["epoch"].append(epoch + 1)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["psnr"].append(avg_psnr)
    history["ssim"].append(avg_ssim)

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.3f}")

# Save results
os.makedirs("results", exist_ok=True)
pd.DataFrame(history).to_csv("results/metrics.csv", index=False)
print("Training complete. Metrics saved to results/metrics.csv ✅")

# Save the trained model
torch.save(model.state_dict(), "results/srcnn_model.pth")
print("Model saved to results/srcnn_model.pth ✅")

