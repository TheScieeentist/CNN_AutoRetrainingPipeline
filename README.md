# YOLO Auto-Retraining Pipeline for Impurity Detection for Waste-Plants
This repository contains my 3,5 week final Capstone-project for the Data Science and AI bootcamp at neue fische. 
It was in colaboration with [WasteAnt](https://wasteant.com/de/ki-basiertes-abfallqualitatsmanagement/).

This repo contains a fully automated pipeline for retraining a YOLO model to detect and classify impurities in waste incineration plants. The system is written in Python and scheduled to run periodically. It evaluates whether sufficient new image data is available to trigger a new training cycle — including data preparation, augmentation, model fine-tuning, validation, and versioning. In a last step, the retrained model is comparted with the current base-model in certain metrics. the old model gets replaced if the performance of the retrained model is improved.

#### *Although, this auto-retraining pipeline was created for detecting waste impurirties, it can easily be adapted to other approaches by simply adapting the /Data/data.yaml file with the required classes.*


## Key Features
 
✅ Automated training cycle based on the number of newly imported images

📁 Structured dataset management with version control and data logs

🔄 Image augmentation with optional class-specific settings

🧠 YOLOv8-based retraining of a base model

📊 Validation and comparison of new vs. current models

🗓️ Scheduled retraining every Tuesday at 13:01 (Berlin time)


## Pipeline Overview

When more than a threshold (default: 1000) of new images are detected, the pipeline performs the following steps:

Train/Validation Split:
Imports new image-label pairs and splits them according to a configurable ratio.

Data Augmentation:
Applies configurable augmentations to selected training images (with optional class-based filtering).

Image Logging:
Logs which images were used, and whether they were augmented, to a CSV file.

Model Retraining:
Fine-tunes the YOLO model using the new dataset and stores the resulting model version.

Validation and Comparison:
Compares the newly trained model to the existing base model using chosen validation metrics (default: mAP@0.5).
If the new model outperforms the current one, it replaces it automatically.


# Directory Structure

```text
/Pipeline
├── Core_Scripts/              # Python core scripts (augmentation, training, etc.)
├── Data/
│   ├── Base/                  # Base training and validation data
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── valid/
│   │       ├── images/
│   │       └── labels/
│   ├── Import/                # Newly imported data waiting for processing
│   │   ├── images/
│   │   └── labels/
│   ├── data.yaml              # YOLO-compatible dataset config
│   └── logs/                  # CSV logs of used images and metadata
├── Model/
│   ├── base_model/            # Current best YOLO model
│   ├── model_versions/        # All past training runs
│   └── model_logs/
│       └── Model_Version_Log.csv  # Performance metrics of each model

```

⚙️ Configuration Parameters
Parameter	Description	Default
Train_Fraction	Portion of import data used for training	0.7
Numb_Aug	Number of augmentations per image	1
Class_Specificity	Class filter for augmentation (0, 1, 2, or None)	2
Portion	Portion of selected images used for augmentation	0.5
Validation_Criteria	Metric for comparing models (map50, etc.)	"map50"
File_Threshold	Minimum number of new images needed to trigger retraining	1000
epochs	Number of fine-tuning epochs	2
imagesize	YOLO input image size	640
batches	Training batch size	16
LearningRate	Learning rate for training	0.001

⏰ Scheduling
The retraining pipeline is automatically scheduled to run every Tuesday at 13:01 CET using APScheduler.

🧪 Requirements
Python 3.8+

PyTorch

YOLOv8 (ultralytics)

APScheduler

Other standard libraries (os, time, logging, pandas, etc.)

🚀 Run Locally
To run the retraining pipeline manually:

bash
Kopieren
Bearbeiten
python trigger.py
To deploy it as a background service with automatic scheduling, ensure trigger.py is running continuously.

📈 Future Work
 Add a web dashboard for training history and performance monitoring

 Implement real-time notifications (e.g., via email or Slack)

 Optimize augmentation pipeline for speed and class balance

🛠 Maintainer
Developed and maintained by [Your Name]
Feel free to contribute or raise issues via GitHub!


