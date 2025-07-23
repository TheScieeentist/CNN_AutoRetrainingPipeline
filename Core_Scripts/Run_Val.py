import os, glob
import cv2
import sys 
import csv
import datetime as dt

import numpy as np
from pathlib import Path
from glob import glob
from ultralytics import YOLO
from common_utils.detection.convertor import xywh2xyxy, xyxyn2xyxy
from common_utils.detection.core import Detections
from common_utils.annotate.core import Annotator

# This function aims to validate the latest model(-weightings) and appends a model-log csv-file with the metrics per class, the name and a time stamp.
# In the lower part is a version, that appends the csv-file with the average metrics of the model

def run_val(Test_Model_Dir: Path, ModelName: str, csv_dir: Path, Model=YOLO):
    
    latest_version = max(glob(os.path.join(Test_Model_Dir, '*/')), key=os.path.getmtime)
    
    model = Model(f"{latest_version}/weights/best.pt")

    ct = dt.datetime.now()

    val_results = model.val(workers = 0) 
    summary = val_results.summary()
    print(summary)
    for i in range(len(summary)):
        summary[i].update({'Model_Name': ModelName, 'Time_Stamp': ct})
    keys = summary[0].keys()

    with open(csv_dir, 'a', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(summary)

    
    # this here is one approach to generate a csv, that append a csv with one row, that contaions the mean box metrics.
    #box = val_results.box

    #with open(csv_dir, 'a', newline='') as f:
    #    writer = csv.writer(f)
    #    writer.writerow([ModelName, box.mp, box.mr, box.map50])'''
    