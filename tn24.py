from ultralytics import YOLO
import glob
import os

# Paths
model_path = r"D:\HIWI\p15\yolov8n_cls_trained.pt"
images_path = r"D:\HIWI\p15\Images"

# Load the trained YOLO model
model = YOLO(model_path)

# Get all images from subdirectories
image_files = glob.glob(os.path.join(images_path, "**", "*.jpg"), recursive=True)

if not image_files:
    print("No .jpg images found in the specified directory.")
else:
    for image in image_files:
        # Run inference
        results = model(image)

        # Get the predicted class and probabilities
        predicted_class = results[0].names[results[0].probs.argmax()]
        probabilities = results[0].probs

        # Print the results
        print(f"Image: {image}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Class Probabilities: {probabilities}\n")
