#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os,glob
import numpy as np
import os
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import cv2
gpu=int(input("Which gpu number you would like to allocate:"))
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
model_name=int(input("Which model you would like to train(TYPE THE NUMBER ONLY LIKE 1)? 1. MOBILE VIT"))
import glob
import pickle
import tensorflow as tf
import argparse
import re
import datetime
import keras
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,Layer,ReLU, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from skimage import data, exposure
from skimage.transform import radon, rescale
from skimage.filters import roberts, sobel, scharr, prewitt
from classification_models.keras import Classifiers
from skimage import feature
import os,glob
import numpy as np
import cv2
import glob
import pickle
import tensorflow as tf
import pickle
import argparse
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
# Display Image
from PIL import Image
# computer vision package to read dataset
import cv2
import os
import datetime
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,ConvLSTM2D,LayerNormalization,GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from skimage import data, exposure
from tensorflow.keras.layers import Layer
from PIL import Image
from numpy import asarray
from sklearn.utils import shuffle
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import layers as L


def inverted_residual_block(inputs, num_filters, strides=1, expansion_ratio=1):
    ## Point-Wise Convolution
    x = L.Conv2D(
        filters=expansion_ratio*inputs.shape[-1],
        kernel_size=1,
        padding="same",
        use_bias=False
    )(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    ## Depth-Wise Convolution
    x = L.DepthwiseConv2D(
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False
    )(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    ## Point-Wise Convolution
    x = L.Conv2D(
        filters=num_filters,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(x)
    x = L.BatchNormalization()(x)

    ## Residual Connection
    if strides == 1 and (inputs.shape == x.shape):
        return L.Add()([inputs, x])
    return x


def mlp(x, mlp_dim, dim, dropout_rate=0.1):
    x = L.Dense(mlp_dim, activation="swish")(x)
    x = L.Dropout(dropout_rate)(x)
    x = L.Dense(dim)(x)
    x = L.Dropout(dropout_rate)(x)
    return x

def transformer_encoder(x, num_heads, dim, mlp_dim):
    skip_1 = x
    x = L.LayerNormalization()(x)
    x = L.MultiHeadAttention(
        num_heads=num_heads, key_dim=dim
    )(x, x)
    x = L.Add()([x, skip_1])

    skip_2 = x
    x = L.LayerNormalization()(x)
    x = mlp(x, mlp_dim, dim)
    x = L.Add()([x, skip_2])

    return x

def mobile_vit_block(inputs, num_filters, dim, patch_size=2, num_layers=1):
    B, H, W, C = inputs.shape

    ## 3x3 conv
    x = L.Conv2D(
        filters=C,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    ## 1x1 conv: d-dimension
    x = L.Conv2D(
        filters=dim,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    ## Reshape x to flattened patches
    P = patch_size*patch_size
    N = int(H*W//P)
    x = L.Reshape((P, N, dim))(x)

    ## Transformr Encoder
    for _ in range(num_layers):
        x = transformer_encoder(x, 1, dim, dim*2)

    ## Reshape
    x = L.Reshape((H, W, dim))(x)

    ## 1x1 conv: C-dimension
    x = L.Conv2D(
        filters=C,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    ## Concatenation
    x = L.Concatenate()([x, inputs])

    ## 3x3 conv
    x = L.Conv2D(
        filters=num_filters,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    return x

def MobileViT(input_shape, num_channels, dim, expansion_ratio, num_layers=[2, 4, 3], num_classes=3):
    ## Input layer
    inputs = L.Input(input_shape)

    ## Stem
    x = L.Conv2D(
        filters=num_channels[0],
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False
    )(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)
    x = inverted_residual_block(x, num_channels[1], strides=1, expansion_ratio=expansion_ratio)

    ## Stage 1
    x = inverted_residual_block(x, num_channels[2], strides=2, expansion_ratio=expansion_ratio)
    x = inverted_residual_block(x, num_channels[3], strides=1, expansion_ratio=expansion_ratio)
    x = inverted_residual_block(x, num_channels[4], strides=1, expansion_ratio=expansion_ratio)

    ## Stage 2
    x = inverted_residual_block(x, num_channels[5], strides=2, expansion_ratio=expansion_ratio)
    x = mobile_vit_block(x, num_channels[6], dim[0], num_layers=num_layers[0])

    ## Stage 3
    x = inverted_residual_block(x, num_channels[7], strides=2, expansion_ratio=expansion_ratio)
    x = mobile_vit_block(x, num_channels[8], dim[1], num_layers=num_layers[1])

    ## Stage 4
    x = inverted_residual_block(x, num_channels[9], strides=2, expansion_ratio=expansion_ratio)
    x = mobile_vit_block(x, num_channels[10], dim[2], num_layers=num_layers[2])
    x = L.Conv2D(
        filters=num_channels[11],
        kernel_size=1,
        padding="same",
        use_bias=False
    )(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    ## Classifier
    x = L.GlobalAveragePooling2D()(x)
    outputs = L.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.models.Model(inputs, outputs)
    return model


def MobileViT_S(input_shape, num_classes):
    num_channels = [16, 32, 64, 64, 64, 96, 144, 128, 192, 160, 240, 640]
    dim = [144, 192, 240]
    expansion_ratio = 4

    return MobileViT(
        input_shape,
        num_channels,
        dim,
        expansion_ratio,
        num_classes=num_classes
    )


   
def data_augmentation(normal_files,covid_files,pneumonia_files):
    aug_normal=[]
    aug_covid=[]
    thresh_hold=7
    aug_pneumonia=[]
    
    #x = tf.keras.preprocessing.image.load_img("/content/IM-0001-0001.jpeg")
    
    datagen=ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,

    )
    #normal
    counter=0
    print("normal files:",len(normal_files))
    print("covid files:",len(covid_files))
    print("pneumonia files:",len(pneumonia_files))
    for location in tqdm(normal_files):
        counter=0

        x = Image.open(location).convert('L')
        x = asarray(x)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        x = clahe.apply(x)
        #x=img_to_array(x)
        x=cv2.resize(x,(256,256),interpolation = cv2.INTER_CUBIC)
        x=x/255.0
        x = np.expand_dims(x, axis=-1) 
        x=x.reshape((1,)+x.shape)
        #x=x/255.0


        for i in datagen.flow(x):
            if counter>=5:
                break
            #i=i/255.0

            #i = cv2.resize(i,(224,224),interpolation = cv2.INTER_CUBIC)
            aug_normal.append(i)
            counter+=1
    #covid
    counter=0
    for location in tqdm(covid_files):
        counter=0
        x = Image.open(location).convert('L')
        x = asarray(x)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        x = clahe.apply(x)
#         x = Image.open(location).convert('L')
#         x = asarray(x)
        x=cv2.resize(x,(256,256),interpolation = cv2.INTER_CUBIC)
        x=x/255.0
        x = np.expand_dims(x, axis=-1) 
        #x=img_to_array(x)
        x=x.reshape((1,)+x.shape)
        #x=x/255.0


        for i in datagen.flow(x):
            if counter>=1:
                break

            aug_covid.append(i)
            counter+=1    
    #pneumonia
    counter=0
    for location in tqdm(pneumonia_files):
        counter=0
        x = Image.open(location).convert('L')
        x = asarray(x)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        x = clahe.apply(x)
        x=cv2.resize(x,(256,256),interpolation = cv2.INTER_CUBIC)
        x=x/255.0
        x = np.expand_dims(x, axis=-1) 
        
        #x=img_to_array(x)
        x=x.reshape((1,)+x.shape)
        #x=x/255.0

        for i in datagen.flow(x):
            if counter>=1:
                break
            #i=i/255.0
            #i = cv2.resize(i,(224,224),interpolation = cv2.INTER_CUBIC)
            aug_pneumonia.append(i)
            counter+=1    

    for ele in normal_files:
        #ele=ele/255.0
        x = Image.open(ele).convert('L')
        x = asarray(x)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        x = clahe.apply(x)
        pic = cv2.resize(x,(256,256),interpolation = cv2.INTER_CUBIC)
        pic=pic/255.0
        aug_normal.append(pic)
    for ele in covid_files:
        #ele=ele/255.0
        x = Image.open(ele).convert('L')
        x = asarray(x)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        x = clahe.apply(x)
        pic = cv2.resize(x,(256,256),interpolation = cv2.INTER_CUBIC)
        pic=pic/255.0
        aug_covid.append(pic)
    for ele in pneumonia_files:
        #ele=ele/255.0
        x = Image.open(ele).convert('L')
        x = asarray(x)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        x = clahe.apply(x)
        pic = cv2.resize(x,(256,256),interpolation = cv2.INTER_CUBIC)
        pic=pic/255.0
        aug_pneumonia.append(pic)
    for i in range(len(aug_normal)):
        aug_normal[i]=aug_normal[i].reshape((256,256))
    
    for i in range(len(aug_covid)):
        aug_covid[i]=aug_covid[i].reshape((256,256))
    for i in range(len(aug_pneumonia)):
        aug_pneumonia[i]=aug_pneumonia[i].reshape((256,256))
    
    print("Normal files after augmentation:",len(aug_normal))
    print("Covid files after augmentation:", len(aug_covid))
    print("Pneumonia files after augmentation:",len(aug_pneumonia))
    return aug_normal,aug_covid,aug_pneumonia

def making_full_data(aug_normal,aug_covid,aug_pneumonia):
    aug_normal=shuffle(aug_normal, random_state=0)
    aug_covid=shuffle(aug_covid,random_state=0)
    aug_pneumonia=shuffle(aug_pneumonia,random_state=0)
    
    aug_normal_labels=[]
    for i in range(len(aug_normal)):
        aug_normal_labels.append(0)
    print(np.shape(aug_normal),np.shape(aug_normal_labels))
    aug_covid_labels=[]
    for i in range(len(aug_covid)):
        aug_covid_labels.append(1)
    print(np.shape(aug_covid),np.shape(aug_covid_labels))
    aug_pneumonia_labels=[]
    for i in range(len(aug_pneumonia)):
        aug_pneumonia_labels.append(2)
    print(np.shape(aug_pneumonia),np.shape(aug_pneumonia_labels))  

    full_data=[]
    full_label=[]
    for i in range(len(aug_normal)):
        full_data.append(aug_normal[i])
        full_label.append(aug_normal_labels[i])
    for i in range(len(aug_covid)):
        full_data.append(aug_covid[i])
        full_label.append(aug_covid_labels[i])
    for i in range(len(aug_pneumonia)):
        full_data.append(aug_pneumonia[i])
        full_label.append(aug_pneumonia_labels[i])
        
    full_data=np.array(full_data)
    full_label=np.array(full_label)
    
    full_data=shuffle(full_data,random_state=0)
    full_label=shuffle(full_label,random_state=0)
    
    return full_data,full_label
"""Inception 2D_CNN Models in Tensorflow-Keras.
References -
Inception_v1 (GoogLeNet): https://arxiv.org/abs/1409.4842 [Going Deeper with Convolutions]
Inception_v2: http://arxiv.org/abs/1512.00567 [Rethinking the Inception Architecture for Computer Vision]
Inception_v3: http://arxiv.org/abs/1512.00567 [Rethinking the Inception Architecture for Computer Vision]
Inception_v4: https://arxiv.org/abs/1602.07261 [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning]
"""
def making_training_and_testing_data(full_data,full_label):
    X_train, X_test, y_train, y_test = train_test_split(full_data, full_label, test_size=0.20, random_state=42)
    
    train_label=[]
    for i in range(len(y_train)):
        if y_train[i]==0:
            train_label.append([0,1,0])
        elif y_train[i]==1:
            train_label.append([1,0,0])
        elif y_train[i]==2:

            train_label.append([0,0,1])

    test_label=[]
    for i in range(len(y_test)):
        if y_test[i]==0:
            test_label.append([0,1,0])
        elif y_test[i]==1:
            test_label.append([1,0,0])
        elif y_test[i]==2:

            test_label.append([0,0,1])
    y_train=np.array(train_label)
    y_test=np.array(test_label)
    
    return X_train,X_test,y_train,y_test
    
def my_plots(history,my_model):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    my_path="training and validation accuracy curve of "+my_model+".png"
    plt.savefig(my_path)
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylim([0, 1])

    #plt.ylim([-3, 3])
    plt.yticks(np.arange(0, 1.1, 0.25))
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    my_path="training and validation loss curve of "+my_model+".png"
    plt.savefig(my_path)
    plt.show()
    
    
if __name__ == '__main__':
    normal_dir = "/home/pranab_2021cs25/Shubham_Amity/ViT_datasets/ViT_dataset/Normal/" #give your normal cases data path here
    #vit_datasets/Dataset_ViT/ViT_dataset/Covid-19
    dir1 = os.path.join(normal_dir,"*.png")
    dir2 = os.path.join(normal_dir,"*.jpeg")
    dir = os.path.join(normal_dir,"*.jpg")
    normal_files = glob.glob(dir)
    normal_1 = glob.glob(dir1)
    normal_2 = glob.glob(dir2)
    normal_files.extend(normal_1)
    normal_files.extend(normal_2)

    normal_dir = "/home/pranab_2021cs25/Shubham_Amity/ViT_datasets/ViT_dataset/Covid-19/"  #give your covid 19 cases data path here
    dir1 = os.path.join(normal_dir,"*.png")
    dir = os.path.join(normal_dir,"*.jpg")
    dir2 = os.path.join(normal_dir,"*.jpeg")
    covid_files = glob.glob(dir)
    covid_files2 = glob.glob(dir2)
    covid_files1 = glob.glob(dir1)
    covid_files.extend(covid_files2)
    covid_files.extend(covid_files1)

    normal_dir = "/home/pranab_2021cs25/Shubham_Amity/ViT_datasets/ViT_dataset/Pneumonia/" #give your pneumonia cases data path here
    dir1 = os.path.join(normal_dir,"*.png")
    dir2 = os.path.join(normal_dir,"*.jpeg")
    dir = os.path.join(normal_dir,"*.jpg")
    pneumonia_files = glob.glob(dir)
    pneumonia_1 = glob.glob(dir1)
    pneumonia_2 = glob.glob(dir2)
    pneumonia_files.extend(pneumonia_1)
    pneumonia_files.extend(pneumonia_2)

    normal_files.sort()
    covid_files.sort()
    pneumonia_files.sort()
    
    aug_normal,aug_covid,aug_pneumonia=data_augmentation(normal_files,covid_files,pneumonia_files)
    
    full_data,full_label=making_full_data(aug_normal,aug_covid,aug_pneumonia)  #getting my full data
    
    X_train,X_test,y_train,y_test= making_training_and_testing_data(full_data,full_label) #dividing full_data into train and test data
    print("Multiplying 3 times")
    train_data=np.stack((X_train,)*3,axis=-1)
    test_data=np.stack((X_test,)*3,axis=-1)
    print("Multiplying done")
    learning_rate = 0.0001
    weight_decay = 0.0001
    batch_size = 16
    num_epochs = 100
    print("now model initialization part")
    if model_name==1:  #IT WILL RUN MOBILE VIT MODEL
        model = MobileViT_S((256, 256, 3), 3)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.CategoricalCrossentropy(from_logits = False), metrics=['accuracy'])

        print("compiling done")
        early_stopping = EarlyStopping(monitor='val_loss', 
            patience=45, 

            min_delta=0.001, 
            mode='min')
        print("model starting fitting")
        history=model.fit(train_data,y_train,epochs=100,validation_data=(test_data, y_test),callbacks=[early_stopping],batch_size=16)
        #history=model.fit(train_data,y_train,epochs=100,validation_data=(test_data, y_test),callbacks=[early_stopping],batch_size=16)
        np.save('my_history_mobvit_with_clahe_fl.npy',history.history)
        my_plots(history,"MOBILE_VIT_with_clahe_fl")
        filename = 'MOBILE_VIT_model_with_clahe_fl.sav'
        pickle.dump(model, open(filename, 'wb'))
        
