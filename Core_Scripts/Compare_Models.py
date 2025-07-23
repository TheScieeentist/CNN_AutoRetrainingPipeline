import os
import cv2
import sys
import torch
import torchvision

import numpy as np
from pathlib import Path
from glob import glob
from ultralytics import YOLO
from common_utils.detection.convertor import xywh2xyxy, xyxyn2xyxy
from common_utils.detection.core import Detections
from common_utils.annotate.core import Annotator


# This function is used to compare the metricies of the lates model(-weights) with the current basemodel(-weights).
# The default metric for the comparison (Validation_Criteria) is the mAP50 as choosen by ultralytics YOLO as parameter itself.
# This function further replaces the current base model(-weights) with the new model(-weights).

def compare_metrics(Current_Model_dir: Path, New_Model_Dir: Path, newModelName:str, Validation_Criteria= "map50", Model=YOLO, newModel=YOLO):
    
    current_weight = f"{Current_Model_dir}/best.pt"
    current_model = Model(current_weight)
    
    latest_version = max(glob(os.path.join(New_Model_Dir, '*/')), key=os.path.getmtime)
    new_weights = f"{latest_version}/weights/best.pt"
    new_model = newModel(new_weights)

    current_val_results = current_model.val(workers = 0)   
    new_val_results = new_model.val(workers = 0)  

    current_box = current_val_results.box
    new_box = new_val_results.box

    print(type(new_box))

    if getattr(new_box, Validation_Criteria) > getattr(current_box, Validation_Criteria):
        os.remove(current_weight)
        src = new_weights
        dst = current_weight
        cmd = f'cp "{src}" "{dst}"'
        os.system(cmd)
        #os.replace(current_weight, new_weights)
        print(f'\n \n The new model performes better. Current baseline model was replaced by the new model {newModelName}\n \n')
    else:
        print("\n \n The current model performed better, thus it remains as basemodel!\n \n")

