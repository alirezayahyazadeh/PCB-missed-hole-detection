import os
import shutil
from ultralytics import YOLO

# Update paths to your dataset
missing_hole_path = r"D:\HIWI\p15\Annotation\Missing_hole"  # Correct this path
open_circuit_path = r"D:\HIWI\p15\Annotation\Open_circuit"  # Correct this path

# Validate paths
if not os.path.exists(missing_hole_path):
    raise FileNotFoundError(f"Missing_hole path not found: {missing_hole_path}")
if not os.path.exists(open_circuit_path):
    raise FileNotFoundError(f"Open_circuit path not found: {open_circuit_path}")

# Step 1: Organize Dataset
base_dir = r"D:\HIWI\p15\YOLO_Dataset"  # Base directory for YOLO dataset
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Class mapping for YOLO labels
classes = {"Missing_hole": 0, "Open_circuit": 1}

# Function to move images and create labels
def organize_data(source_folder, class_name, images_dir, labels_dir):
    image_count = 0
    for image_file in os.listdir(source_folder):
        if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            source_image = os.path.join(source_folder, image_file)
            target_image = os.path.join(images_dir, image_file)
            print(f"Copying {source_image} to {target_image}")
            shutil.copy(source_image, target_image)
            image_count += 1

            # Create a dummy label
            label_file = os.path.join(labels_dir, os.path.splitext(image_file)[0] + ".txt")
            with open(label_file, "w") as f:
                f.write(f"{classes[class_name]} 0.5 0.5 1.0 1.0\n")
    return image_count

# Organize data for each class
missing_hole_count = organize_data(missing_hole_path, "Missing_hole", images_dir, labels_dir)
open_circuit_count = organize_data(open_circuit_path, "Open_circuit", images_dir, labels_dir)
print("Dataset organized successfully.")
print(f"Images directory contains: {missing_hole_count + open_circuit_count} images")
print(f"Labels directory contains: {len(os.listdir(labels_dir))} labels")
