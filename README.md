# YOLO Auto-Retraining Pipeline for Impurity Detection for Waste-Plants
This repository contains my 3,5 week final Capstone-project for the Data Science and AI bootcamp at neue fische. 
It was in colaboration with [WasteAnt](https://wasteant.com/de/ki-basiertes-abfallqualitatsmanagement/).

This repo contains a fully automated pipeline for retraining a YOLO model to detect and classify impurities in waste incineration plants. The system is written in Python and scheduled to run periodically. It evaluates whether sufficient new image data is available to trigger a new training cycle — including data preparation, augmentation, model fine-tuning, validation, and versioning. In a last step, the retrained model is comparted with the current base-model in certain metrics. The old model gets replaced if the performance of the retrained model is improved.

#### *Although, this auto-retraining pipeline was created for detecting waste impurirties, it can easily be adapted to other approaches by simply adapting the /Data/data.yaml file with the required classes.*


## Key Features
 
✅ Automated training cycle

📁 Structured dataset management with data logs

🔄 Image augmentation with optional class-specific settings

🧠 YOLOv11-based retraining of a base model

📊 Validation and comparison of retrained vs. current base models

🗓️ Scheduleable retraining


## Pipeline Overview

When more than a threshold (default: 1000) of new images are detected in the import-fodler, the pipeline performs the following steps:

Train/Validation Split:
Splits new image-label pairs according to a configurable ratio and distributes them into the corresponding base-folders.

Data Augmentation:
Applies configurable augmentations to selected training images (with optional class-based filtering).

Image Logging:
Logs which images were used for training/validation, and whether they were augmented, to a CSV file.

Model Retraining:
Re-trains/fine-tunes the current baseline-model using the new trainings-dataset and stores the resulting model version.

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
│   │   ├── 20250722_retraining_Image_Log
├── Model/
│   ├── base_model/            # Current best YOLO model
│   │   ├── best.pt
│   ├── model_versions/        # All past training runs
│   │   ├── 20250722_YOLO_TestRun_v1
│   └── model_logs/
│       └── Model_Version_Log.csv  # Performance metrics of each model

```

## ⚙️ Configuration Parameters

| Parameter            | Description                                                                 | Default       |
|----------------------|-----------------------------------------------------------------------------|---------------|
| `Train_Fraction`     | Fraction of imported images used for training                               | `0.7`         |
| `Numb_Aug`           | Number of augmentations per image                                           | `2`           |
| `Class_Specificity`  | Class filter for augmentation (`0`, `1`, `2`, or `None`)                    | `None`        |
| `Portion`            | Portion of selected images used for augmentation                            | `0.5`         |
| `Validation_Criteria`| Metric to evaluate models                                                   | `"map50"`     |
| `File_Threshold`     | Minimum number of new images required to trigger retraining                 | `1000`        |
| `epochs`             | Number of training epochs for fine-tuning                                   | `200`         |
| `imagesize`          | Input image size for YOLO                                                   | `640`         |
| `batches`            | Batch size for training                                                     | `16`          |
| `LearningRate`       | Learning rate for training                                                  | no default!   |
| `TrainingRunName`    | Name used to label this training run       




## 🧪 Requirements: 
* Python 3.8+

* PyTorch

* YOLOv8 (ultralytics)

* APScheduler

* Albumentations

* Other standard libraries (os, time, logging, pandas, os, sys etc.)


