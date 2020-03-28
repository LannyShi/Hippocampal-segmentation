#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Sun Feb 16 14:02:23 2020

@author: Itai Muzhingi

"""

import os

import nibabel as nib

import numpy as np

from skimage.transform import resize
from unet import *
from network import *
import matplotlib.pyplot as plt
import random
import cv2
from single_channel import *
from multi_channels import *


#tf.enable_eager_execution()

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#gpu_options = tf.GPUOptions(allow_growth = True)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def filenames():
    images = os.listdir("./Task04_Hippocampus/imagesTr/")
    labels = os.listdir("./Task04_Hippocampus/labelsTr/")
    images.sort()
    labels.sort()
    return images, labels

image_list, label_list = filenames()
parent_dir = os.getcwd()
#training_images, training_labels = get_M_C_1_images(image_list, label_list)   #（32,32,3) (32,32,1)
training_images, training_labels = get_M_C_2_images(image_list, label_list) #（32,32,3) (32,32,1)
#training_images, training_labels = get_S_C_images(image_list, label_list)  #（32,32,1) (32,32,1)

print(training_images.shape, 'training1_images')
print(training_labels.shape, 'training1_labels')

normalized_images = training_images / 255.0

img_train = normalized_images[200:1700, :, :, :]
#seg_train = training_labels[208:1700, :, :, :]
seg_train = training_labels[200:1700, :, :, :]

img_val = normalized_images[:200, :, :, :]
seg_val = training_labels[:200, :, :, :]
#img_val = normalized_images[:208, :, :, :]
#seg_val = labels_distort[:208, :, :, :]
img_val_2 = normalized_images[1700:, :, :, :]
seg_val_2 = training_labels[1700:, :, :, :]
#img_val_2 = normalized_images[3260:, :, :, :]
#seg_val_2 = training_labels[3260:, :, :, :]
print(img_val_2.shape) #（120,32,32,1）
print(seg_val_2.shape) #（120,32,32,1）
print(img_train.shape) #（1492,32,32,1）
print(seg_train.shape) #（1492,32,32,1）
#index = [i for i in range(len(img_train))]
#random.shuffle(index)
#img_train = img_train[index]
#seg_train = seg_train[index]
#print(img_train.shape) #（1492,32,32,1）
#print(seg_train.shape) #（1492,32,32,1）

# train
#model = get_unet(input_dim=(32, 32, 1), output_dim=(32, 32, 1), num_output_classes=1)
#model = get_att_unet(input_dim=(32, 32, 1), output_dim=(32, 32, 1), num_output_classes=1)
#model = get_HR_Att_unet_1(input_dim=(32, 32, 1), output_dim=(32, 32, 1), num_output_classes=1)
model = get_HR_unet()
history = model.fit(x=img_train, y=seg_train, batch_size=32, epochs=200, verbose=1, validation_data=(img_val, seg_val))
#history = model.fit(x=img_train, y=seg_train, batch_size=32, epochs=300, verbose=1, validation_split=0.2, shuffle=True)

# loss和dice曲线
plt.figure()
dice = history.history['dice_coef']
val_dice = history.history['val_dice_coef']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(dice))
plt.plot(epochs, dice, 'bo', label='training dice')
plt.plot(epochs, val_dice, 'b', label='validation dice')
plt.title('training and validation dice')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('training and validation loss')
plt.legend()
plt.show()

# save model
os.chdir(parent_dir)
model.save_weights('model_weights_best_1.h5')

# test
def multi_channels(image1, image2, image3):
    image1 = np.mat(image1)
    image2 = np.mat(image2)
    image3 = np.mat(image3)
    img = cv2.merge([image1, image2, image3])
    return img

os.chdir("./Task04_Hippocampus/imagesTs/")
test_array = []
test_list = os.listdir()
for i in range(len(test_list)):
    testing_image = nib.load(test_list[i])
    testing_image = resize(testing_image.get_fdata(), (32, 32, 32))
    image_slice1 = testing_image[15, :, :]
    image_slice2 = testing_image[16, :, :]
    image_slice3 = testing_image[17, :, 16]
    image_slice = multi_channels(image_slice1, image_slice2, image_slice3)
    #test_array.append(image_slice[:, :, np.newaxis])
    test_array.append(image_slice)
final_test = np.array(test_array)
final_test = final_test / 255.0
prediction = model.predict(final_test)
print(prediction.shape)

for i in range(len(prediction)):
    plt.imshow(np.squeeze(final_test[i]), cmap='gray')
    plt.show()    # test_images
    image = np.squeeze(prediction[i], axis=2)
    plt.imshow(np.rint(np.clip(image, 0, 1)))
    plt.show()   # test_results（0,1）
    plt.imshow(image, cmap='jet')
    plt.show()   # test_results（heat map）


plt.plot(history.history['loss'], label='Training Loss', linestyle='--')
plt.title('Training Loss', fontweight='bold', fontsize=18)
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch Number', fontweight='bold', fontsize=15)
plt.ylabel('Loss', fontweight='bold', fontsize=15)
plt.legend(loc='upper right')
plt.grid(True, color='#999999', linestyle='--', alpha=0.2)
plt.show()

#plt.plot(history.history['acc'], color='black', label='Training Accuracy')
plt.plot(history.history['dice_coef'], color='black', label='Training Accuracy')
plt.title('Training Accuracy', fontweight='bold', fontsize=18)
#plt.plot(history.history['val_acc'], color='red', label='Validation Accuracy', linestyle='--')
plt.plot(history.history['val_dice_coef'], color='red', label='Validation Accuracy', linestyle='--')
plt.xlabel('Epoch Number', fontweight='bold', fontsize=15)
plt.ylabel('Accuracy', fontweight='bold', fontsize=15)
plt.legend(loc='lower right')
plt.grid(True, color='#999999', linestyle='--', alpha=0.2)
plt.show()


