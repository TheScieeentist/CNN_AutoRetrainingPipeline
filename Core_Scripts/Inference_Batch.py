import os
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from tqdm import tqdm

# 1. Load YOLO model
#model_path = "/home/appuser/src/Models/YOLO_m_raw3/weights/best.pt" #yolo_m_raw
#model_path = "/home/appuser/src/Models/YOLO_s_raw/weights/best.pt" #yolo_s_raw
#model_path = "/home/appuser/src/Models/YOLO_n_raw3/weights/best.pt" #yolo_n_raw

# YOLO aug
#model_path = "/home/appuser/src/Models/YOLO_s_aug/weights/best.pt" #yolo_s_aug
#model_path = "/home/appuser/src/Models/YOLO_n_aug/weights/best.pt" #yolo_n_aug

#WasteImpurity
#model_path = "/home/appuser/src/Models/Wastelmp_raw3/weights/best.pt" #wasteimp_raw
model_path = "/home/appuser/src/Models/WasteImp_Aug2/weights/best.pt" #wasteimp_aug

model = YOLO(model_path).to("cuda")

# 2. Warm-up
dummy_input = torch.randn(1, 3, 640, 640).to("cuda") / 255.0
for _ in range(10):
    _ = model(dummy_input)

# 3. Folder with images
image_dir = "/home/appuser/src/data/combined_data/train_val_split/train/augmented_plus/images"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg.jpg")]

# 4. Measure inference times over all images
inference_times = []
postprocess_times = []

with torch.no_grad():
    for filename in tqdm(image_files, desc="Running inference"):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image_tensor = torch.tensor(image, dtype=torch.float32).to("cuda")

        results = model(image_tensor, verbose=False)
        inference_times.append(results[0].speed["inference"])
        postprocess_times.append(results[0].speed["postprocess"])

# 5. Results
mean_inf = np.mean(inference_times)
std_inf = np.std(inference_times)
mean_post = np.mean(postprocess_times)
std_post = np.std(postprocess_times)
fps = 1000 / mean_inf

print(f"Images evaluated:   {len(image_files)}")
print(f"Inference Time:     {mean_inf:.2f} ms ± {std_inf:.2f} ms")
print(f"Postprocess Time:   {mean_post:.2f} ms ± {std_post:.2f} ms")
print(f"Estimated FPS:      {fps:.2f}")
