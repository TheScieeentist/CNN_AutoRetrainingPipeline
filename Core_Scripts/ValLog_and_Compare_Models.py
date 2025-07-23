import os
import datetime as dt
import csv

from pathlib import Path
from glob import glob
from ultralytics import YOLO



# This function is used to compare the metricies of the lates model(-weights) with the current basemodel(-weights).
# The default metric for the comparison (Validation_Criteria) is the mAP50 as choosen by ultralytics YOLO as parameter itself.
# This function further replaces the current base model(-weights) with the new model(-weights).

def val_and_compare(New_Model_Dir: Path,Current_Model_dir: Path,  csv_dir: Path, newModelName:str, Validation_Criteria= "map50", Model=YOLO, newModel=YOLO):
        
    latest_version = max(glob(os.path.join(New_Model_Dir, '*/')), key=os.path.getmtime)
    new_weights = f"{latest_version}/weights/best.pt"
    new_model = newModel(new_weights)

    ct = dt.datetime.now()

    new_val_results = new_model.val(workers = 0)  
    summary = new_val_results.summary()

    print(summary)
    for i in range(len(summary)):
        summary[i].update({'Model_Name': newModelName, 'Time_Stamp': ct})
    keys = summary[0].keys()

    with open(csv_dir, 'a', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(summary)


    current_weight = f"{Current_Model_dir}/best.pt"
    current_model = Model(current_weight)
    current_val_results = current_model.val(workers = 0)   

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
        print(f"\n \n The current model performed better, thus it remains as basemodel! \nThe new model {newModelName} can be found under {New_Model_Dir} . \n \n")

