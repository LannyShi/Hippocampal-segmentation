import nibabel as nib

import numpy as np

from skimage.transform import resize
from network import *
import matplotlib.pyplot as plt
import random
import cv2
from data_processing import *
import os

def filenames():
    images = os.listdir("data_path")
    labels = os.listdir("label_path")
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
os.chdir(parent_dir)
#new_model = get_unet(input_dim=(32, 32, 1), output_dim=(32, 32, 1), num_output_classes=1)
new_model = get_HR_unet()
new_model.summary()
new_model.load_weights('model_weights_best_1.h5')
pred_val = new_model.predict(img_val_2)
for i in range(len(pred_val)):
    plt.imshow(np.squeeze(img_val_2[i]), cmap='gray')
    plt.show()   # val_images
    plt.imshow(np.squeeze(seg_val_2[i]), cmap='gray')
    plt.show()    # val_labels
    new_pred = np.squeeze(pred_val[i], axis=2)
    plt.imshow(np.rint(np.clip(new_pred, 0, 1)))
    plt.show()    # val_results（0,1）
    plt.imshow(new_pred, cmap='jet')
    plt.show()     # val_results

def dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / (im1.sum() + im2.sum())
list_dice = []
for i in range(len(pred_val)):
    val_label = np.squeeze(seg_val_2[i])
    val_label = np.array(val_label, dtype='f')
    pred_label = np.squeeze(pred_val[i], axis=2)
    pred_label = np.array(pred_label, dtype='f')
    list_dice.append(dice(val_label, pred_label))

results = np.array(list)
print(resuts)

plt.hist(list_dice, color='brown', edgecolor='black', linewidth=1.2)
plt.title('Dice Score Distribution on Test Data', fontweight='bold', fontsize=18)
plt.xlabel('Dice Score', fontweight='bold', fontsize=15)
plt.ylabel('Frequency', fontweight='bold', fontsize=15)
plt.grid(True, color='#999999', linestyle='--', alpha=0.2)
plt.show()