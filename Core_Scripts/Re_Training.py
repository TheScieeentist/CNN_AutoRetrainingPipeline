# loading necessary packages and libraries....c+p from run_NVy:
import os
import cv2
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from glob import glob
from ultralytics import YOLO
from common_utils.annotate.core import Annotator
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
import time

from tqdm import tqdm
import torch.nn as nn

# This function runs the training with the new trainings-data including the new augmentations.
# It generates a model name based on the timestamp, the model used (by default YOLO) and a version
# count based on teh number of folders inside the destination folder.
# It further retunrs the new modelname as a string for further use.

def fine_tuning(BaseModelWeights_dir: Path, yaml_path: Path, ModelDestPath: Path, RunName: str, Model='YOLov11n', epochs=200, imagesize=640, batches=16, LearningRate=0.001):

    Version_count = int(len(os.listdir(ModelDestPath))+1)
    timestr = time.strftime("%Y%m%d_")
    NewModelName = f"{timestr}_{RunName}_v{Version_count}"

    torch.cuda.is_available()

    current_weight = f"{BaseModelWeights_dir}/best.pt"
    model = Model(current_weight) 

    model.train(    data=str(yaml_path), 
                    epochs= epochs, imgsz = imagesize, 
                    batch=batches, lr0=LearningRate, 
                    project=ModelDestPath, 
                    name=NewModelName, 
                    workers=0)
    
    return NewModelName   