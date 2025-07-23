import os
import sys
import csv
import torch
from pathlib import Path
import time



# function für listen
def image_list_to_csv(data_base_directory: Path, csv_file_dir: Path):
    timestr = time.strftime("%Y%m%d_%H%M_")
    with open(f"{csv_file_dir}/{timestr}Train_Im_Log", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "train/val/test", "augmentation" ])  # Header

        for subfolder in os.listdir(data_base_directory):

            folder_path = os.path.join(f"{data_base_directory}/{subfolder}/images")
            if not os.path.isdir(folder_path):
                continue 

            label = subfolder

            for image_name in os.listdir(folder_path):
                if image_name.endswith(".jpg"):
                    if image_name.startswith("aug_"):
                        writer.writerow([image_name, label, 'augmented'])
                    else:
                        writer.writerow([image_name, label, 'original'])
                