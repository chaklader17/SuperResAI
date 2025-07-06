# preprocessing.py

from PIL import Image
import os
import cv2

# Step 1: Resize raw images to 1024x1024
def resize_to_1024(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        try:
            img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
            img = img.resize((1024, 1024))
            img.save(os.path.join(output_dir, fname))
        except Exception as e:
            print(f"Skipping {fname}: {e}")

# Step 2: Create low-res version (64x64 → 1024x1024)
def degrade_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        path = os.path.join(input_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"Could not load {fname}")
            continue

        small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(small, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(output_dir, fname), degraded)

# Run both steps
if __name__ == "__main__":
    resize_to_1024("data/raw", "data/high_res")
    degrade_images("data/high_res", "data/low_res")
