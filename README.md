PCB Defect Classification Using YOLOv8
This project focuses on classifying defects in printed circuit boards (PCBs) using the YOLOv8 model. It identifies two types of defects: Missing Hole and Open Circuit. The repository provides scripts to preprocess data, train a YOLOv8 model, and make predictions.

Table of Contents
Overview
Dataset
Setup and Installation
Workflow
Scripts Description
Results
Contributing
License
Overview
This project automates the detection and classification of PCB defects using YOLOv8. Key features include:

Data preprocessing and conversion to YOLO-compatible format.
Training the YOLOv8 model on the processed dataset.
Evaluating and making predictions on new data.
Exporting the trained model to ONNX format for broader compatibility.
Dataset
The dataset used for this project includes images of PCBs with annotations in XML format (VOC format). The classes are:

Missing Hole
Open Circuit
Dataset Structure
The dataset is organized into the following structure:



Images/
├── Missing_hole/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Open_circuit/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
Annotation/
├── Missing_hole/
│   ├── image1.xml
│   ├── image2.xml
│   └── ...
├── Open_circuit/
    ├── image1.xml
    ├── image2.xml
    └── ...
Setup and Installation
Clone the repository:


git clone (https://github.com/alirezayahyazadeh/PCB-missed-hole-detection)
cd pcb-defect-classification
Install required dependencies:

pip install -r requirements.txt
Ensure the dataset is placed in the correct directory structure as described above.

Workflow
Dataset Preparation: Converts XML annotations to YOLO format and splits the data into training and validation sets.
Training: Trains the YOLOv8 model on the processed dataset.
Evaluation: Validates the trained model on the validation set.
Inference: Runs inference on unseen images to predict defect classes.
Scripts Description
t13.py
This script handles the entire process from dataset preparation to model training and evaluation.

Key Functions:
prepare_dataset: Converts VOC XML annotations to YOLO format.
validate_dataset: Ensures all images have corresponding annotations.
split_dataset: Splits the data into training and validation sets.
create_data_yaml: Generates the data.yaml file for YOLO training.
train_yolo_model: Trains the YOLOv8 model with configurable parameters.
evaluate_model: Evaluates the trained YOLO model.
save_predictions: Saves predictions from the trained model.
How to Run:

python t13.py

This script performs inference on images using the trained YOLOv8 model.

Key Steps:
Loads the trained model (yolov8n_cls_trained.pt).
Scans the image directory for inference.
Outputs predicted classes and probabilities for each image.
How to Run:

python tn24.py
Results
After training the YOLOv8 model, the following outputs are generated:

Trained Model: Saved as yolov8n_cls_trained.pt.
Predictions: Stored in the runs/detect directory.
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

