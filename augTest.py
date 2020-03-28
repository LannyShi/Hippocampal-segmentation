from utilsDb import distort_img
# from utilsDb import distort
from PIL import Image

import nibabel as nib
# import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

scanID = "hippocampus_001"

scanPath = "./Task04_Hippocampus/imagesTr/" + scanID + ".nii.gz"
labelPath = "./Task04_Hippocampus/labelsTr/" + scanID + ".nii.gz"

img = nib.load(scanPath)
scan = img.get_fdata()
print(scan.shape)
print(type(scan))
#scan = img.get_data()
scan0 = scan[20]
print(scan0.shape)

mask = nib.load(labelPath)
maskData = mask.get_fdata()
#maskData = mask.get_data()
mask0 = maskData[20]


maskCat = to_categorical(mask0)
mask1 = maskCat[:, :, 0]
print(mask1.shape)
mask0 = 1-mask1

data = [scan0, mask0]
scan_distorted, mask_distorted = distort_img(data)

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(scan0)
plt.title("MRI Scan")


plt.subplot(2, 2, 2)
plt.imshow(mask0)
plt.title("Label")



plt.subplot(2, 2, 3)
plt.imshow(scan_distorted)
plt.title("Distorted Scan")

mask_distorted = np.round(mask_distorted)
# mask_distorted = np.ceil(mask_distorted)

plt.subplot(2, 2, 4)
plt.imshow(mask_distorted)
plt.title("Distorted Label")
plt.show()

#https://towardsdatascience.com/full-stack-deep-learning-steps-and-tools-a21eda6227b1