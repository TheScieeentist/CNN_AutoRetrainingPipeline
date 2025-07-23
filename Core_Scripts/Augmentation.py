import albumentations as A
import random
import cv2
import numpy as np
import torch
from tqdm import tqdm
import os, glob
import sys
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from common_utils.detection.core import Detections
from pathlib import Path

# Function for augmenting either a certain percentage of all trainings-images or all trainings-images containing a certain class.
# The trainings-image and -lable folder will contain the augmented images afterwards. These "aug_*" images and lables will be removed as first thing 
# with every run of the function and new augmentations will begenerated

def augmentation(SOURCE_IMAGE_DIR: Path, SOURCE_LABEL_DIR: Path, Rseed = 42, Numb_Aug=1, Class_Specificity=None, Portion=0.4):

    #print(f"Zielverzeichnis Bilder: {SOURCE_IMAGE_DIR}")
    #print(f"Zielverzeichnis Labels: {SOURCE_LABEL_DIR}")

    RSEED = Rseed

    # removing of pre-existing augmented images and related labels
    for filename in glob.glob(f"{SOURCE_IMAGE_DIR}/aug_*"):
        os.remove(filename)
    
    for filename in glob.glob(f"{SOURCE_LABEL_DIR}/aug_*"):
        os.remove(filename)

    # liste der augmentation-combinations:
    aug_combi_1 =   A.Compose([ A.HorizontalFlip(p=1.0),
                                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=1.0)], 
                                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
                                )


    aug_combi_2 =   A.Compose([ A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=25, val_shift_limit=15, p=1.0),
                                A.RandomBrightnessContrast(p=1.0)],
                                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
                                )
                        

    aug_combi_3 = A.Compose([   A.Rotate(limit=15, p=1.0),  
                                A.MotionBlur(blur_limit=5, p=1.0),],
                                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
                                )


    aug_combi_4 = A.Compose([   A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=1.0),
                                A.ChromaticAberration(primary_distortion_limit=0.05, secondary_distortion_limit=0.1, 
                                                      mode='green_purple', interpolation=cv2.INTER_LINEAR,p=1.0)],
                                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
                                )


    aug_combi_5 = A.Compose([   A.Equalize(mask_params=['bboxes'], p=1.0),
                                A.VerticalFlip(p=1.0)],
                                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
                                )        


    # choosing one of the augmentation-combinations for each augmentation of the image
    random_transform = A.OneOf([aug_combi_1, aug_combi_2, aug_combi_3, aug_combi_4, aug_combi_5])


    # Funktions from Tannous Geagea needed for creating bboxed-versions of the augmented images
    #def xywh2xyxy(xywh):
        
        #Convert bounding box coordinates from (x, y, width, height) format to (xmin, ymin, xmax, ymax) format.

        #This function assumes (x, y) as the center of the bounding box and calculates 
        #the coordinates of the top-left corner (xmin, ymin) and the bottom-right corner (xmax, ymax).

        #Parameters:
        #- xywh (Tuple[float, float, float, float]): A tuple representing the bounding box in (x, y, width, height) format.

        #Returns:
        #- Tuple[float, float, float, float]: A tuple representing the bounding box in (xmin, ymin, xmax, ymax) format.
        
    #    x, y, w, h = xywh
    #    return (x - w/2, y - h/2, x + w/2, y + h/2)

    # Funktions from Tannous Geagea needed for creating bboxed-versions of the augmented images
    #def xyxyn2xyxy(xyxyn, image_shape):
        
        #Convert bounding box coordinates from normalized format to pixel format.

        #This function converts the normalized bounding box coordinates back to pixel format. 
        #The normalized coordinates (xmin_n, ymin_n, xmax_n, ymax_n), represented as fractions 
        #of the image's width or height, are scaled back to the pixel dimensions of the image.

        #Parameters:
        #- xyxyn (tuple): A tuple of four floats (xmin_n, ymin_n, xmax_n, ymax_n) representing the normalized bounding box coordinates.
        #- image_shape (tuple): A tuple of two integers (height, width) representing the dimensions of the image.

        #Returns:
        #- tuple: A tuple of four integers (xmin, ymin, xmax, ymax) representing the bounding box coordinates in pixel format.
       
    #    xmin, ymin, xmax, ymax = xyxyn
    #    return (int(xmin * image_shape[1]), int(ymin * image_shape[0]), int(xmax * image_shape[1]), int(ymax * image_shape[0]))


    # Funktions from Tannous Geagea needed for creating bboxed-versions of the augmented images
    #def box_label(im, box, line_width=3, label='', cls_id=0, color=None, txt_color=None):

    #    """Add one xyxy box to image with label."""
        
    #    input_shape = im.shape
    #    lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

        # cv2
    #    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    #    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    #    if label:
    #        tf = max(lw - 1, 1)  # font thickness
    #        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    #        outside = p1[1] - h >= 3
    #        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    #        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
    #        cv2.putText(im,
    #                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
    #                    0,
    #                    lw / 3,
    #                    txt_color,
    #                    thickness=tf,
    #                    lineType=cv2.LINE_AA)
    #    return im

    # reading of yolo_labels:
    def read_yolo_label(path):
        bboxes = []
        class_labels = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                bbox = list(map(float, parts[1:]))
                bboxes.append(bbox)
                class_labels.append(class_id)
        return bboxes, class_labels

    

    # setting colors for bboxes if bboxed images are created:
    #colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

    # excluding images without annotations:
    all_images = []
    
    for image_file in os.listdir(SOURCE_IMAGE_DIR):
        if not image_file.endswith(".jpg.jpg"):
            continue
        label_file = image_file.replace(".jpg.jpg", ".jpg.txt")
        label_path = Path(SOURCE_LABEL_DIR)/label_file
        if label_path.exists() and os.stat(label_path).st_size > 0:
            all_images.append(image_file)

    # extracting images containing a specific class, if defined previously:    
    #if Class_Specificity == None:
    #    selected_images = set(random.sample(all_images, int(Portion * len(all_images))))
    #else:
    #    selected_images = all_images

    selected_images = set(random.sample(all_images, int(Portion * len(all_images))))
        
    # calling/processing individual images from the previous selection for augmentation:
    for image_file in tqdm(selected_images):

        if not image_file.endswith(".jpg.jpg"):
            continue

        image_path = os.path.join(SOURCE_IMAGE_DIR, image_file)
        label_path = os.path.join(SOURCE_LABEL_DIR, image_file.replace(".jpg.jpg", ".jpg.txt"))

        print(f"Verarbeite: {image_file}")

        image = cv2.imread(image_path)
        bboxes, class_labels = read_yolo_label(label_path)


        #print(bboxes,type(class_labels), class_labels)
        
        # skipping images if they do not contain annotations from the desired class:
        if Class_Specificity != None:
            if not Class_Specificity in class_labels: 
                continue

        # augmenting each image as often as defined with Numb_Aug:
        for i in range(Numb_Aug): 
            try:
                transform = random_transform
                augmented = transform(  image=image,
                                        bboxes=bboxes,
                                        class_labels=class_labels        
                                        )

            except Exception as e:
                print(f"Augmentation failed: {e}")
                continue

            aug_image = augmented['image']

            # relevant for non-YOLO models or if A.Compose contains ToTensorV2:
            #if not isinstance(aug_image, np.ndarray): # not necessary if ToTensorV2 is not in aug_lists!!!
            #    aug_image = np.array(aug_image)
            #    aug_image = np.transpose(aug_image, (1, 2, 0))
            #    print(aug_image.shape)

            aug_bboxes = augmented['bboxes']
            aug_class_labels = augmented["class_labels"]

            # if images with ploted bboxes are needed, these lines (and all hashtaged functions from Tannous Geagea) are needed:
            #aug_image_boxed = aug_image.copy()

            #print(aug_bboxes)
            #print(class_labels)

            #for j,x in enumerate(aug_bboxes):
            #    classID = int(aug_class_labels[j])
            #    xyxy = xywh2xyxy(x)
            #    print(xyxy)
            #    n_xyxy = xyxyn2xyxy(xyxy,aug_image.shape)   
            #    print(n_xyxy) 
            #    aug_image_boxed = box_label(aug_image_boxed, n_xyxy, color=colors[classID])
            
            #print(f"Speichere augmentiertes Bild #{i} mit {len(aug_bboxes)} Zielobjekten")
            #print(f"type aug_image: {type(aug_image)}")
            #print(f"aug_image==None? {aug_image is None}")
            #print(f"type aug_image: {type(aug_image)}")
            #print(f"shape aug_image: {aug_image.shape}")

            # transforming labels 
            classes = np.expand_dims(aug_class_labels, axis=1)
            if len(aug_bboxes):
                lines = [("%g " * len(line)).rstrip() %tuple(line) + "\n" for line in np.hstack((classes.astype(int), aug_bboxes))]

            with open(f"{SOURCE_LABEL_DIR}/aug_{image_file.replace('.jpg.jpg', '_')}{i}.txt", "w") as f:
                f.writelines(lines)

            #saving the augmented images
            cv2.imwrite(os.path.join(SOURCE_IMAGE_DIR, f"aug_{image_file.replace('.jpg.jpg', '_')}{i}.jpg"), aug_image)
            
            # this is also needed if bboxed images are wanted:
            #cv2.imwrite(os.path.join(f"{SOURCE_IMAGE_DIR}/bboxed_images", f"boxed_aug_{image_file.replace('.jpg.jpg', '_')}.jpg"), aug_image_boxed)

