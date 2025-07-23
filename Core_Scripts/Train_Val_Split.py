import os, glob
import random
import shutil
from pathlib import Path

# Ursprung

def test_val_split(Image_Path: Path, Label_Path: Path, Dest_Path: Path, RSEED=42, Train_Fraction=0.7): 

    # Zielordner:
    for split in ["train", "valid"]:
        os.makedirs(os.path.join(Dest_Path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(Dest_Path, split, "labels"), exist_ok=True)

    # Bilder sammeln 
    imgs_list = os.listdir(Image_Path)

    # Zufällig mischen und Reproduzierbarkeit gewährleisten
    random.seed(RSEED)
    random.shuffle(imgs_list)

    # Aufteilen stardardmäßig: 70 % train, 30 % valid
    n = len(imgs_list)
    train_size = int(n * Train_Fraction)

    splits = {
        "train": imgs_list[:train_size],
        "valid": imgs_list[train_size:]
    }

    # Kopieren von Bild + Label
    for split, file_list in splits.items():
        for img_file in file_list:
            label_file = os.path.splitext(img_file)[0] + ".txt"

            # Quelle
            img_src = os.path.join(Image_Path, img_file)
            lbl_src = os.path.join(Label_Path, label_file)

            # Zielpfade 
            img_dst = os.path.join(Dest_Path, split, "images", img_file)
            lbl_dst = os.path.join(Dest_Path, split, "labels", label_file)

            # Bild kopieren
            shutil.copy(img_src, img_dst)

            # Label kopieren 
            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, lbl_dst)
    

    # removing of images/labels in the import folder:
    for filename in glob.glob(f"{Image_Path}/*"):
        os.remove(filename)
    
    for filename in glob.glob(f"{Label_Path}/*"):
        os.remove(filename)


    print("\n \n \n \n Train_Valid_Split done! \n \n \n \n")
