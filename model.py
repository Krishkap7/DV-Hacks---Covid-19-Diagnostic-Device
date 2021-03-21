"""Import libraries"""

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

# functions
def display_image(img):
    '''
    function to show a picture 
    '''
    if type(img) == 'str':
        img = plt.imread(img)
    plt.imshow(img, cmap = 'gray')
    plt.title('Example X-Ray scan')
    plt.grid(False)
    plt.axis('off')
    plt.show()

"""Dataset download"""

!git clone https://github.com/Krishkap7/DV-Hacks---Covid-19-Diagnostic-Device

"""Scan Healthy vs. Covid images and 1-hot encode labels

"""

datasetPath = 'DV-Hacks---Covid-19-Diagnostic-Device/real_data'

# read all files from path
print("loading images...")
imagePaths = list(paths.list_images(datasetPath))

data = []

# select no. of classes to be loaded for Healthy, Bacterial Pneumonia, and Viral Pneumonia patients 
numHealthyImages = 364

i = 0 # initialize all 3 counters to decr. the no. of images loaded 
for imagePath in imagePaths:
    # extract class label from file name
    imageLabel = imagePath.split(os.path.sep)[-2]

    if imageLabel == 'healthy' and i < numHealthyImages:
      data.append((imagePath,imageLabel))
      i += 1

    elif imageLabel == 'covid':
      data.append((imagePath,imageLabel))

print('\n creating a DataFrame...')
# retrieve pandas dataframe from our data in the list 
data = pd.DataFrame(data, columns=['image', 'label'],index=None)

# randomize data
data = data.sample(frac=1.).reset_index(drop=True)
print(data.head(10))

tmp_img = plt.imread(data["image"][0])
display_image(tmp_img)
del tmp_img

# do 1-hot encoding on labels
print('\n one-hot encoding...')
print('\tExample scan label:', data["label"][0])

data["label"] = LabelEncoder().fit_transform(data["label"])
print('\tOne-hot encoded label: ',data["label"][0])
print('\n updating data DataFrame...')
print(data.head(10))

"""Visualize example X-ray scans

"""

# choose the no. of examples to show in order to confirm that the covid and healthy ones are same no.
numExamples = 5
healthyExample = (data[data['label']==0]['image'].iloc[30:30+numExamples]).tolist()
covidExample = (data[data['label']==1]['image'].iloc[5:5+numExamples]).tolist()

samples = covidExample + healthyExample
del covidExample, healthyExample

color = 'black'

# plot data 
f, ax = plt.subplots(2, numExamples, figsize=(40,15))
for i in range(numExamples * 2):
    img = cv2.imread(samples[i])
    img = cv2.resize(img, (224, 224))
    if i < numExamples:
        ax[0, i % numExamples].imshow(img, cmap='gray')
        ax[0, i % sample_num].set_title("Covid-19", fontsize = 30, color = color)
    else:
        ax[1, i % numExamples].imshow(img, cmap='gray')
        ax[1, i % numExamples].set_title("Healthy", fontsize = 30, color = color)

    ax[i // numExamples, i%numExamples].axis('off')
    ax[i // numExamples, i % numExamples].set_aspect('auto')

plt.tight_layout()
plt.show()
