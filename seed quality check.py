# -*- coding: utf-8 -*-


!nvidia-smi


import os
HOME = os.getcwd()
print(HOME)



# Pip install method (recommended)

!pip install ultralytics==8.2.103 -q

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

from IPython.display import display, Image



# Commented out IPython magic to ensure Python compatibility.
!mkdir -p {HOME}/datasets
# %cd {HOME}/datasets

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="EUzCaEyrniNKumik94Wa")
project = rf.workspace("aathika-lil").project("seed-quality-a7hsc")
version = project.version(2)
dataset = version.download("yolov8")

"""## Custom Training"""

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}

!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=25 imgsz=800 plots=True

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/confusion_matrix.png', width=600)

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/results.png', width=600)

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/val_batch0_pred.jpg', width=600)

"""## Validate Custom Model"""

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}

!yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml

"""## Inference with Custom Model"""

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}
!yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True



import glob
from IPython.display import Image, display

# Define the base path where the folders are located
base_path = '/content/runs/detect/'

# List all directories that start with 'predict' in the base path
subfolders = [os.path.join(base_path, d) for d in os.listdir(base_path)
              if os.path.isdir(os.path.join(base_path, d)) and d.startswith('predict')]

# Find the latest folder by modification time
latest_folder = max(subfolders, key=os.path.getmtime)

image_paths = glob.glob(f'{latest_folder}/*.jpg')[:3]

# Display each image
for image_path in image_paths:
    display(Image(filename=image_path, width=600))
    print("\n")

## Deploy model on Roboflow


project.version(dataset.version).deploy(model_type="yolov8", model_path=f"{HOME}/runs/detect/train/")




model = project.version(dataset.version).model
assert model, "Model deployment is still loading"

# Choose a random test image
import os, random
test_set_loc = dataset.location + "/test/images/"
random_test_image = random.choice(os.listdir(test_set_loc))
print("running inference on " + random_test_image)

pred = model.predict(test_set_loc + random_test_image, confidence=40, overlap=30).json()
pred

