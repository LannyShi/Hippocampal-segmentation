import os
import nibabel as nib
import numpy as np
import math
from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt

def filenames():
    images = os.listdir("./Task04_Hippocampus/imagesTr/")
    labels = os.listdir("./Task04_Hippocampus/labelsTr/")
    images.sort()
    labels.sort()
    return images, labels

image_list, label_list = filenames()
parent_dir = os.getcwd()

def multi_channels(image1, image2, image3):
    image1 = np.mat(image1)
    image2 = np.mat(image2)
    image3 = np.mat(image3)
    img = cv2.merge([image1, image2, image3])
    return img

def get_M_C_1_images(image_list, label_list):
    shape = (32, 32, 32)
    image_array = []
    label_array = []
    for i in range(len(image_list)):
        if image_list[i] == label_list[i]:
            os.chdir("./Task04_Hippocampus/imagesTr/")
            image = nib.load(image_list[i])
            resampled_image = resize(image.get_fdata(), shape)
            for i in range(7):
                image_slice_i_x = resampled_image[i+13, :, :]
                image_slice_i_y = resampled_image[:, i+13, :]
                image_slice_i_z = resampled_image[:, :, i+13]
                image_slice_i = multi_channels(image_slice_i_x, image_slice_i_y, image_slice_i_z)
                image_array.append(image_slice_i)
            os.chdir(parent_dir)
            os.chdir("./Task04_Hippocampus/labelsTr/")
            label = nib.load(label_list[i])
            resampled_label = np.rint(np.clip(resize(label.get_fdata(), shape), 0, 1))
            for k in range(7):
                label_slice_k_x = resampled_label[k+13, :, :]
                label_slice_k_y = resampled_label[:, k+13, :]
                label_slice_k_z = resampled_label[:, :, k+13]
                label_slice_k = multi_channels(label_slice_k_x, label_slice_k_y, label_slice_k_z)
                #print(label_slice_k.shape)
                label_array.append(label_slice_k)
            os.chdir(parent_dir)
                #os.chdir(parent_dir)
    return np.array(image_array), np.array(label_array)

def get_M_C_2_images(image_list, label_list):
    shape = (32, 32, 32)
    image_1_array = []
    label_1_array = []
    for i in range(len(image_list)):
        if image_list[i] == label_list[i]:
            os.chdir("./Task04_Hippocampus/imagesTr/")
            image = nib.load(image_list[i])
            resampled_image = resize(image.get_fdata(), shape)
            for i in range(7):
                image_slice_i_0 = resampled_image[i+12, :, :]
                image_slice_i_1 = resampled_image[i+13, :, :]
                image_slice_i_2 = resampled_image[i+14, :, :]
                image_slice_i = multi_channels(image_slice_i_0, image_slice_i_1, image_slice_i_2)
                image_1_array.append(image_slice_i)
            os.chdir(parent_dir)
            os.chdir("./Task04_Hippocampus/labelsTr/")
            label = nib.load(label_list[i])
            resampled_label = np.rint(np.clip(resize(label.get_fdata(), shape), 0, 1))
            for k in range(7):
                label_slice_k = resampled_label[k+13, :, :]
                label_1_array.append(label_slice_k[:, :, np.newaxis])
            os.chdir(parent_dir)
    return np.array(image_1_array), np.array(label_1_array)

def get_S_C_images(image_list, label_list):
    shape = (32, 32, 32)
    image_2_array = []
    label_2_array = []
    for i in range(len(image_list)):
        if image_list[i] == label_list[i]:
            os.chdir("./Task04_Hippocampus/imagesTr/")
            image = nib.load(image_list[i])
            resampled_image = resize(image.get_fdata(), shape)
            for i in range(7):
                image_slice_i = resampled_image[i+13, :, :]
                image_2_array.append(image_slice_i[:, :, np.newaxis])
            os.chdir(parent_dir)
            os.chdir("./Task04_Hippocampus/labelsTr/")
            label = nib.load(label_list[i])
            resampled_label = np.rint(np.clip(resize(label.get_fdata(), shape), 0, 1))
            for k in range(7):
                label_slice_k = resampled_label[k+13, :, :]
                label_2_array.append(label_slice_k[:, :, np.newaxis])
            os.chdir(parent_dir)
    return np.array(image_2_array), np.array(label_2_array)







