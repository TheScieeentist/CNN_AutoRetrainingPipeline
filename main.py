import os
import time
import logging
#import sys
#import csv
#import torch
from pathlib import Path
#from tqdm import tqdm
#from torch.utils.data import DataLoader
from ultralytics import YOLO
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import timezone

from Pipeline.Core_Scripts.Train_Val_Split import test_val_split
from Pipeline.Core_Scripts.Augmentation import augmentation
from Pipeline.Core_Scripts.Save_To_List import image_list_to_csv
from Pipeline.Core_Scripts.Re_Training import fine_tuning
from Pipeline.Core_Scripts.Run_Val import run_val
from Pipeline.Core_Scripts.Compare_Models import compare_metrics
from Pipeline.Core_Scripts.ValLog_and_Compare_Models import val_and_compare



DATA_MAIN_DIR = '/home/appuser/src/Pipeline/Data'

CORE_SCRIPTS_DIR = '/home/appuser/src/Pipeline/Core_Scripts'


MODEL_MAIN_DIR = '/home/appuser/src/Pipeline/Model'
MODEL_BASEMODEL_DIR = '/home/appuser/src/Pipeline/Model/base_model'
MODEL_LOGS_DIR = '/home/appuser/src/Pipeline/Model/model_logs/Model_Version_Log.csv'
MODEL_VERSIONS_DIR = '/home/appuser/src/Pipeline/Model/model_versions'

DATA_LOGS_DIR = '/home/appuser/src/Pipeline/Data/logs'

BASE_DATA_DIR = '/home/appuser/src/Pipeline/Data/Base'
BASE_YAML_DIR = '/home/appuser/src/Pipeline/Data/data.yaml'

BASE_TRAIN_DIR = '/home/appuser/src/Pipeline/Data/Base/train'
BASE_TRAIN_IM_DIR = '/home/appuser/src/Pipeline/Data/Base/train/images'
BASE_TRAIN_LB_DIR = '/home/appuser/src/Pipeline/Data/Base/train/labels'

BASE_VAL_DIR = '/home/appuser/src/Pipeline/Data/Base/valid'
BASE_VAL_IM_DIR = '/home/appuser/src/Pipeline/Data/Base/valid/images'
BASE_VAL_LB_DIR = '/home/appuser/src/Pipeline/Data/Base/valid/labels'

IMPORT_DATA_DIR = '/home/appuser/src/Pipeline/Data/Import'
IMPORT_IM_DIR = '/home/appuser/src/Pipeline/Data/Import/images'
IMPORT_LB_DIR = '/home/appuser/src/Pipeline/Data/Import/labels'

RSEED = 42
Train_Fraction = 0.7    # Fraction of images added to training-dataset from new import!
Numb_Aug = 1            # Number of augmentations per selected image. Maximum is 5!
Class_Specificity = 2   # Class selection for image augmentation, current options are 0, 1, 2 or None
Portion = 0.5           # Defines the fraction of selected images used for augmentation. 
                        # Caution: this only applies if Class_Specificity = None!

Model = YOLO
ModelWeights = "/home/appuser/src/Pipeline/Model/base_model/WasteImpurityMultiClass_agr_bunker_V5.pt"
Validation_Criteria = "map50" # map50 is default, other options are 



# condition for main():
def file_amount(directory: Path):
    files = os.listdir(directory)
    return len(files)


def main(BASE_DATA_DIR=BASE_DATA_DIR, 
         IMPORT_IM_DIR=IMPORT_IM_DIR, 
         IMPORT_LB_DIR=IMPORT_LB_DIR, 
         RSEED=RSEED, 
         Train_Fraction=Train_Fraction,
         BASE_TRAIN_DIR=BASE_DATA_DIR,
         BASE_TRAIN_IM_DIR=BASE_TRAIN_IM_DIR,
         BASE_TRAIN_LB_DIR=BASE_TRAIN_LB_DIR,
         Numb_Aug=Numb_Aug, Class_Specificity=Class_Specificity, 
         Portion=Portion,
         DATA_LOGS_DIR=DATA_LOGS_DIR,
         YAML_DIR=BASE_YAML_DIR,
         Model=Model,
         MODEL_VERSIONS_DIR=MODEL_VERSIONS_DIR,
         TrainingRunName='Test_Training',
         epochs=2, 
         imagesize=640, 
         batches=16, 
         LearningRate=0.001,
         MODEL_LOGS_DIR=MODEL_LOGS_DIR,
         MODEL_BASEMODEL_DIR=MODEL_BASEMODEL_DIR,
         Validation_Criteria=Validation_Criteria,
         File_Threshold=1000):
    
    if file_amount(IMPORT_IM_DIR) >= File_Threshold: # File threshold is 1000 by default
        # 1. Step: train_val_split:
        test_val_split(Image_Path=IMPORT_IM_DIR, Label_Path=IMPORT_LB_DIR, Dest_Path=BASE_DATA_DIR, RSEED=RSEED, Train_Fraction=Train_Fraction)

        # 2. Step: augmentation:
        augmentation(SOURCE_IMAGE_DIR=BASE_TRAIN_IM_DIR, SOURCE_LABEL_DIR=BASE_TRAIN_LB_DIR, Rseed=RSEED, Numb_Aug=Numb_Aug, Class_Specificity=Class_Specificity, Portion=Portion)

        # 3. Step: printing the imagenames, their use and if augmented to a table:
        image_list_to_csv(data_base_directory=BASE_TRAIN_DIR, csv_file_dir=DATA_LOGS_DIR)

        # 4. Step: fine-tuning/re-training of the current model:
        nmn = fine_tuning(Model=Model, BaseModelWeights_dir=MODEL_BASEMODEL_DIR,yaml_path=YAML_DIR, ModelDestPath=MODEL_VERSIONS_DIR, RunName=TrainingRunName, epochs=epochs, imagesize=imagesize, batches=batches, LearningRate=LearningRate)
            # default values for the re-training: epochs=200, imagesize=640, batches=16, LearningRate=0.001

        # 5. Step: validate (latest) model and compare with validation of current model:
        val_and_compare(New_Model_Dir=MODEL_VERSIONS_DIR, Current_Model_dir=MODEL_BASEMODEL_DIR, csv_dir=MODEL_LOGS_DIR ,newModelName=nmn, Validation_Criteria=Validation_Criteria)
        
        
        '''# 5. Step: validate (latest) model:
        run_val(Test_Model_Dir=MODEL_VERSIONS_DIR, ModelName=nmn, csv_dir=MODEL_LOGS_DIR)

        # 6. Step: comparing metrics of new and current model. If new model performs better, current model will be replaced by new one:
        compare_metrics(Current_Model_dir=MODEL_BASEMODEL_DIR, New_Model_Dir= MODEL_VERSIONS_DIR,newModelName=nmn Validation_Criteria=Validation_Criteria)
            # default settings for comparing the metrics: Validation_Criteria= "mAP50", Model=YOLO, newModel=YOLO'''
    else:
        print('\n \n There are not enough new images in the Import-Folder for a new training!\n \n')



# setting up the scheduler using timezone to set the timezone to german time stadard:
scheduler = BackgroundScheduler(timezone=timezone('Europe/Berlin'))
scheduler.start()

# adjusting the scheduler to run the main() script at every _________ at ____:
scheduler.add_job(main, 'cron', day_of_week='tue', hour=13, minute=1)

logging.basicConfig()
logging.getLogger('apscheduler').setLevel(logging.DEBUG)

# this part defines the interruptions of the trigger.py, so that it runs in the background 
# as long as no interruption is called with the keyboard in the terminal:
try:
    while True:
        time.sleep(1)
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown()

#if __name__ == "__main__":
#   main()


    # time inference in die compare metrics + run val -> no easy solution!


        # Tipp von Thomas: 
    #       Alle Bilder pre-prozessen vor dem training mit: Farben-verbesser bzw. Albumentations Equalizer 
    #       bzw. GIMP Farben-verbesserung und farb-spreizung.
    #       ggf. ChannelShuffel raus holen
    #       alpha verändern, elastic, color-channel droppen
    #           -> Thomas-GitHub Repo zu augmentation nachgucken!