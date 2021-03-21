# -*- coding: utf-8 -*-

"""Import Necesarry Libraries"""

try:
#     %tensorflow_version 2.x
#     %load_ext tensorboard   
except Exception:
  pass

import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from imutils import paths
import seaborn as sns
import imgaug as aug
import pandas as pd
import numpy as np
import datetime
import h5py
import cv2
import os

# helper functions
def display_image(img):
    """
    function to show a picture of a chest x-ray scan
    """
    if type(img) == 'str':
        img = plt.imread(img)
    plt.imshow(img, cmap = 'gray')
    plt.title('Example X-Ray scan')
    plt.grid(False)
    plt.axis('off')
    plt.show()
 """Dataset download"""

!git clone https://github.com/Krishkap7/DV-Hacks---Covid-19-Diagnostic-Device

"""Scan Normal vs. Covid images"""

data_path = 'DV-Hacks---Covid-19-Diagnostic-Device/real_data'

# read all the files in the path
print("loading images...")
imagePaths = list(paths.list_images(data_path))

data = []

i = 0 #initialize the counters and add all 3 in the model 
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    if label == 'normal' and i < num_normal_images:
      data.append((imagePath,label))
      i += 1

    elif label == 'covid':
      data.append((imagePath,label))
print('\n Displaying a dataframe...')

# Create a pandas dataframe
data = pd.DataFrame(data, columns=['image', 'label'],index=None)

# Shuffle the data 
data = data.sample(frac=1.).reset_index(drop=True)
print(data.head(10))

current_image = plt.imread(data["image"][0])
display_image(current_image)
del current_image

# perform one-hot encoding on the labels
print('\n creating a one-hot encoding')
print('\tExample to display:', data["label"][0])

data["label"] = LabelEncoder().fit_transform(data["label"])
print('\tOne-hot encoded label: ',data["label"][0])
print('\n re creating the dataframe')
print(data.head(10))


"""Visualise example X-ray scans"""

# select the number of examples to display - make sure that the covid and normal ones are the same number
num_examples = 5
normal_example = (data[data['label']==0]['image'].iloc[30:30+num_examples]).tolist()
covid_example = (data[data['label']==1]['image'].iloc[5:5+num_examples]).tolist()

samples = covid_example + normal_example
del covid_example, normal_example

color = 'black'
# Plot the data 
f, ax = plt.subplots(2, num_examples, figsize=(40,15))
for i in range(num_examples * 2):
    img = cv2.imread(samples[i])
    img = cv2.resize(img, (224, 224))
    if i < num_examples:
        ax[0, i % num_examples].imshow(img, cmap='gray')
        ax[0, i % sample_num].set_title("Covid-19", fontsize = 30, color = color)
    else:
        ax[1, i % num_examples].imshow(img, cmap='gray')
        ax[1, i % num_examples].set_title("Normal", fontsize = 30, color = color)

    ax[i // num_examples, i%num_examples].axis('off')
    ax[i // num_examples, i % num_examples].set_aspect('auto')

plt.tight_layout()
plt.show()
