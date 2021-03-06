"""Import libraries"""


try:
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

# No of data points 
numHealthyImages = 364

i = 0
for imagePath in imagePaths:
    # find the name of the class through the path tracker
    imageLabel = imagePath.split(os.path.sep)[-2]

    if imageLabel == 'healthy' and i < numHealthyImages:
      data.append((imagePath,imageLabel))
      i += 1

    elif imageLabel == 'covid':
      data.append((imagePath,imageLabel))

# retrieve pandas dataframe
data = pd.DataFrame(data, columns=['image', 'label'],index=None)

# randomize the dataset
data = data.sample(frac=1.).reset_index(drop=True)
print(data.head(10))

current_image = plt.imread(data["image"][0])
display_image(current_image)
del current_image

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
"""Split data for training and testing"""

train_data, test_data = train_test_split(data, test_size=0.20,shuffle=True,random_state=12)
train_data, valid_data = train_test_split(train_data, test_size=0.20, random_state=12)

print('\nNumber of training pairs: ', len(train_data))
print('Number of validation pairs: ', len(valid_data))
print('Number of testing pairs: ', len(test_data))

data['label'].value_counts()

# Get the counts for each class
cases_count = data['label'].value_counts()
print('\t----- Entire Dataset ------')
print('\t\t', cases_count.ravel())

cases_count_tr = train_data['label'].value_counts()
cases_count_val = valid_data['label'].value_counts()
cases_count_tst = test_data['label'].value_counts()
print('\n-- Train Set -- -- Validation Set -- -- Test Set --')
print('  ', cases_count_tr.ravel(),'\t     ',cases_count_val.ravel(), '\t        ', cases_count_tst.ravel(), '\n\n')

# Plot the results 
plt.figure(figsize=(12,10))
sns.barplot(x=cases_count.index, y= cases_count.values, palette=sns.cubehelix_palette(4, start=2.5, rot=0.6))
plt.title('Frequency of Each Class in the Dataset', fontsize=14, color='w') 
plt.xlabel('Case type', fontsize=12, color='w')
plt.ylabel('Count', fontsize=12, color='w')
plt.xticks(range(len(cases_count.index)), ['Normal (0)', 'Covid-19 (1)'], color='w')
plt.yticks(color='w')
plt.show()

def get_arrays(df):
    images, labels = [], []

    img_paths = df.iloc[:,0].values # extract image paths from DataFrame
    labels_ = df.iloc[:,1].values # extract labels from DataFrame

    for i,path in enumerate(img_paths):
        # load the image, swap color channels, and resize it to be a fixed 224x224 pixels while ignoring aspect ratio
        image = cv2.imread(path)

        # check if it's grayscale
        if image.shape[2]==1:
            print(image.shape[2])
            image = np.dstack([image, image, image])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        image = image / 255.0 # Normalize images to range [0,1]
        # print('pre: ', labels_[i])
        encoded_label = tf.keras.utils.to_categorical(labels_[i], num_classes=2)
        # print('encoded: ',encoded_label )
        images.append(image)
        labels.append(encoded_label)

    return np.array(images), np.array(labels)

trainX, trainY = get_arrays(train_data)
validX, validY = get_arrays(valid_data)
testX, testY = get_arrays(test_data)

# calculating class weights from trainset for class imabalance
print(cases_count_tr)
covid_pneumonia_count = cases_count_tr.ravel()[1]
normal_count = cases_count_tr.ravel()[0]

class_weights = {0: 1.0, 1: normal_count / covid_count} 

print('\nclass weights: ', class_weights)

def get_arrays(df):
    images, labels = [], []

    img_paths = df.iloc[:,0].values # extract image paths from DataFrame
    labels_ = df.iloc[:,1].values # extract labels from DataFrame

    for i,path in enumerate(img_paths):
        # load the image, swap color channels, and resize it to be a fixed 224x224 pixels while ignoring aspect ratio
        image = cv2.imread(path)

        # check if it's grayscale
        if image.shape[2]==1:
            print(image.shape[2])
            image = np.dstack([image, image, image])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        image = image / 255.0 # Normalize images to range [0,1]
        # print('pre: ', labels_[i])
        encoded_label = tf.keras.utils.to_categorical(labels_[i], num_classes=2)
        # print('encoded: ',encoded_label )
        images.append(image)
        labels.append(encoded_label)

    return np.array(images), np.array(labels)

trainX, trainY = get_arrays(train_data)
validX, validY = get_arrays(valid_data)
testX, testY = get_arrays(test_data)

# calculating class weights from trainset for class imabalance
print(cases_count_tr)
covid_pneumonia_count = cases_count_tr.ravel()[1]
normal_count = cases_count_tr.ravel()[0]

class_weights = {0: 1.0, 1: normal_count / covid_count} 

print('\nclass weights: ', class_weights)

"""Train on a VGG-19 Model"""

def build_model(startModel):

      # Training models
      newModel = startModel.output
      newModel = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(newModel)
      newModel = tf.keras.layers.Flatten(name="flatten")(newModel)
      newModel = tf.keras.layers.Dense(64, activation="relu")(newModel)
      newModel = tf.keras.layers.Dropout(0.5)(newModel)
      newModel = tf.keras.layers.Dense(2, activation="sigmoid")(newModel)

      # Stack layers
      model = tf.keras.models.Model(inputs=startModel.input, outputs=newModel, name='Covid19_Diagnostic')

      return model

startModel = classifier = tf.keras.applications.VGG19(weights="imagenet", include_top=False,
            input_tensor = tf.keras.layers.Input(shape=(512,512,3)))
model = build_model(startModel)

model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer='adam', metrics=["accuracy"])

# initialize the training data augmentation object
trained_model = tf.keras.preprocessing.image.ImageDataGenerator(
    # rescale= 1 / 255.0,
    rotation_range=15,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    fill_mode="nearest")

  
confirmed_model = tf.keras.preprocessing.image.ImageDataGenerator()


early_stopping = tf.keras.callbacks.EarlyStopping(
monitor='val_accuracy', 
verbose=1,
patience=15,
mode='max',
restore_best_weights=True
)

reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=3, verbose=0, 
                                                                    mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint('VGG19_weights.h5', 
                                                      monitor='val_accuracy', verbose=0, save_best_only=True, 
                                                      save_weights_only=True, mode='max')


callbacks = [early_stopping, reduce_on_plateau, model_checkpoint]

batch_size = 15
epochs = 100
results = model.fit(
    train_datagen.flow(trainX, trainY, batch_size=batch_size),
    steps_per_epoch=len(trainX) // batch_size,
    validation_data=valid_datagen.flow(validX, validY, batch_size=batch_size),
    validation_steps=len(validX) // batch_size,
    class_weight = class_weights,
    epochs=epochs,
    callbacks=callbacks
)
"""Make final predictions"""

# Make predictions on test data
predIdxs = model.predict(testX, batch_size=batch_size)

# Find the predicted labels 
predIdxs = np.argmax(predIdxs, axis=1)
print('\nThis is our models prediction: (Covid=1, Normal=0):')
print(predIdxs)
# print(testY)
trueIdxs = np.argmax(testY, axis=1)
print(trueIdxs)

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
"""Split data for training and testing"""

train_data, test_data = train_test_split(data, test_size=0.20,shuffle=True,random_state=12)
train_data, valid_data = train_test_split(train_data, test_size=0.20, random_state=12)

print('\nNumber of training pairs: ', len(train_data))
print('Number of validation pairs: ', len(valid_data))
print('Number of testing pairs: ', len(test_data))

data['label'].value_counts()

# Get the counts for each class
cases_count = data['label'].value_counts()
print('\t----- Entire Dataset ------')
print('\t\t', cases_count.ravel())

cases_count_tr = train_data['label'].value_counts()
cases_count_val = valid_data['label'].value_counts()
cases_count_tst = test_data['label'].value_counts()
print('\n-- Train Set -- -- Validation Set -- -- Test Set --')
print('  ', cases_count_tr.ravel(),'\t     ',cases_count_val.ravel(), '\t        ', cases_count_tst.ravel(), '\n\n')

# Plot the results 
plt.figure(figsize=(12,10))
sns.barplot(x=cases_count.index, y= cases_count.values, palette=sns.cubehelix_palette(4, start=2.5, rot=0.6))
plt.title('Frequency of Each Class in the Dataset', fontsize=14, color='w') 
plt.xlabel('Case type', fontsize=12, color='w')
plt.ylabel('Count', fontsize=12, color='w')
plt.xticks(range(len(cases_count.index)), ['Normal (0)', 'Covid-19 (1)'], color='w')
plt.yticks(color='w')
plt.show()

def get_arrays(df):
    images, labels = [], []

    img_paths = df.iloc[:,0].values # extract image paths from DataFrame
    labels_ = df.iloc[:,1].values # extract labels from DataFrame

    for i,path in enumerate(img_paths):
        # load the image, swap color channels, and resize it to be a fixed 224x224 pixels while ignoring aspect ratio
        image = cv2.imread(path)

        # check if it's grayscale
        if image.shape[2]==1:
            print(image.shape[2])
            image = np.dstack([image, image, image])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        image = image / 255.0 # Normalize images to range [0,1]
        # print('pre: ', labels_[i])
        encoded_label = tf.keras.utils.to_categorical(labels_[i], num_classes=2)
        # print('encoded: ',encoded_label )
        images.append(image)
        labels.append(encoded_label)

    return np.array(images), np.array(labels)

trainX, trainY = get_arrays(train_data)
validX, validY = get_arrays(valid_data)
testX, testY = get_arrays(test_data)

# calculating class weights from trainset for class imabalance
print(cases_count_tr)
covid_pneumonia_count = cases_count_tr.ravel()[1]
normal_count = cases_count_tr.ravel()[0]

class_weights = {0: 1.0, 1: normal_count / covid_count} 

print('\nclass weights: ', class_weights)

def get_arrays(df):
    images, labels = [], []

    img_paths = df.iloc[:,0].values # extract image paths from DataFrame
    labels_ = df.iloc[:,1].values # extract labels from DataFrame

    for i,path in enumerate(img_paths):
        # load the image, swap color channels, and resize it to be a fixed 224x224 pixels while ignoring aspect ratio
        image = cv2.imread(path)

        # check if it's grayscale
        if image.shape[2]==1:
            print(image.shape[2])
            image = np.dstack([image, image, image])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        image = image / 255.0 # Normalize images to range [0,1]
        # print('pre: ', labels_[i])
        encoded_label = tf.keras.utils.to_categorical(labels_[i], num_classes=2)
        # print('encoded: ',encoded_label )
        images.append(image)
        labels.append(encoded_label)

    return np.array(images), np.array(labels)

trainX, trainY = get_arrays(train_data)
validX, validY = get_arrays(valid_data)
testX, testY = get_arrays(test_data)

# calculating class weights from trainset for class imabalance
print(cases_count_tr)
covid_pneumonia_count = cases_count_tr.ravel()[1]
normal_count = cases_count_tr.ravel()[0]

class_weights = {0: 1.0, 1: normal_count / covid_count} 

print('\nclass weights: ', class_weights)

"""Train on a VGG-19 Model"""

def build_model(startModel):

      # Training models
      newModel = startModel.output
      newModel = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(newModel)
      newModel = tf.keras.layers.Flatten(name="flatten")(newModel)
      newModel = tf.keras.layers.Dense(64, activation="relu")(newModel)
      newModel = tf.keras.layers.Dropout(0.5)(newModel)
      newModel = tf.keras.layers.Dense(2, activation="sigmoid")(newModel)

      # Stack layers
      model = tf.keras.models.Model(inputs=startModel.input, outputs=newModel, name='Covid19_Diagnostic')

      return model

startModel = classifier = tf.keras.applications.VGG19(weights="imagenet", include_top=False,
            input_tensor = tf.keras.layers.Input(shape=(512,512,3)))
model = build_model(startModel)

model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer='adam', metrics=["accuracy"])

# initialize the training data augmentation object
trained_model = tf.keras.preprocessing.image.ImageDataGenerator(
    # rescale= 1 / 255.0,
    rotation_range=15,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    fill_mode="nearest")

  
confirmed_model = tf.keras.preprocessing.image.ImageDataGenerator()


early_stopping = tf.keras.callbacks.EarlyStopping(
monitor='val_accuracy', 
verbose=1,
patience=15,
mode='max',
restore_best_weights=True
)

reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=3, verbose=0, 
                                                                    mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint('VGG19_weights.h5', 
                                                      monitor='val_accuracy', verbose=0, save_best_only=True, 
                                                      save_weights_only=True, mode='max')


callbacks = [early_stopping, reduce_on_plateau, model_checkpoint]

batch_size = 15
epochs = 100
results = model.fit(
    train_datagen.flow(trainX, trainY, batch_size=batch_size),
    steps_per_epoch=len(trainX) // batch_size,
    validation_data=valid_datagen.flow(validX, validY, batch_size=batch_size),
    validation_steps=len(validX) // batch_size,
    class_weight = class_weights,
    epochs=epochs,
    callbacks=callbacks
)
"""Make final predictions"""

# Make predictions on test data
predIdxs = model.predict(testX, batch_size=batch_size)

# Find the predicted labels 
predIdxs = np.argmax(predIdxs, axis=1)
print('\nThis is our models prediction: (Covid=1, Normal=0):')
print(predIdxs)
# print(testY)
trueIdxs = np.argmax(testY, axis=1)
print(trueIdxs)
