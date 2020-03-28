import nibabel as nib
import glob

# import numpy as np
import h5py
from utilsDb import distort_img

# Utility Imports

from utilsDb import capScan
from utilsDb import normalize
from utilsDb import return2DslicesAsList
from utilsDb import resizeStack

from keras.utils import to_categorical

# ## global parameters ###

size = [35, 50, 35]
xsize, ysize, zsize = size
thresh = 4000
plane = "xy"

scanName = "hippocampus_001.nii.gz"

scanPath = "./Task04_Hippocampus/imagesTr/*.nii.gz" # + scanName
labelPath = "./Task04_Hippocampus/labelsTr/*.nii.gz" #  + scanName

scans = sorted(glob.glob(scanPath))
labels = sorted(glob.glob(labelPath))

# scan = scans[0]
# label = labels[0]

# I need to iterate through all the images to get the number of images

nimages = 0

for scan in scans:
    img = nib.load(scan)
    imgData = img.get_fdata()
    if plane == "yz":
        n = imgData.shape[0]
    if plane == "zx":
        n = imgData.shape[1]
    if plane == "xy":
        n = imgData.shape[2]
    nimages = nimages + n

# ##################### for a single image ##################################



# todo: store all the images of a scan inlcuding mask in an h5 file.

# todo: store the masks in categorical format

# todo: apply compression and formatting to dataset

mult = 4

nimages = nimages * (mult + 1)

if plane == "yz":
    shape = (ysize, zsize)
if plane == "zx":
    shape = (zsize, xsize)
if plane == "xy":
    shape = (xsize, ysize)

shape = (nimages,) + shape

h5file = h5py.File("./data/testfile.h5", "w")
h5file.create_dataset("image", shape)
h5file.create_dataset("mask", shape)

def storeSingleImage(didx,scanPath,labelPath):
    img = nib.load(scanPath)
    imgData = img.get_fdata()

    mask = nib.load(labelPath)
    maskData = mask.get_fdata()

    capped = capScan(imgData, thresh)
    slices = return2DslicesAsList(capped, plane)
    scanResized = resizeStack(slices, plane, size)

    masks = return2DslicesAsList(maskData, plane)
    maskResized = resizeStack(masks, plane, size)

    for (scan, mask) in zip(scanResized, maskResized):
        maskCat = to_categorical(mask)
        mask1 = maskCat[:, :, 0]
        mask = 1 - mask1
        data = [scan, mask]
        scans = [scan]
        masks = [mask]
        for i in range(mult):
            scan_distorted, mask_distorted = distort_img(data)
            scans.append(scan_distorted)
            masks.append(mask_distorted)
        for (img, mask) in zip(scans, masks):
            h5file["image"][didx] = normalize(img, thresh)
            h5file["mask"][didx] = mask
            didx = didx + 1
    return didx

didx = 0

for (scanPath,labelPath) in zip(scans,labels):
    print("Processing: " + scanPath)
    didx = storeSingleImage(didx, scanPath, labelPath)


h5file.close()

#didx = storeSingleImage(didx, scans[0], labels[0])

'''

img = nib.load(scanPath)
imgData = img.get_fdata()

mask = nib.load(labelPath)
maskData = mask.get_fdata()

capped = capScan(imgData, thresh)
slices = return2DslicesAsList(capped, plane)
scanResized = resizeStack(slices, plane, size)

masks = return2DslicesAsList(maskData, plane)
maskResized = resizeStack(masks, plane, size)

for (scan, mask) in zip(scanResized, maskResized):
    maskCat = to_categorical(mask)
    mask1 = maskCat[:, :, 0]
    mask = 1 - mask1
    data = [scan, mask]
    scans = [scan]
    masks = [mask]
    for i in range(mult):
        scan_distorted, mask_distorted = distort_img(data)
        scans.append(scan_distorted)
        masks.append(mask_distorted)
    for (img, mask) in zip(scans, masks):
        h5file["image"][didx] = normalize(img, thresh)
        h5file["mask"][didx] = mask
        didx = didx + 1

'''



"""
didx = 0

for img in scanResized:
    h5file["image"][didx] = normalize(img, thresh)
    didx = didx + 1

didx = 0
for masks in maskResized:
    h5file["mask"][didx] = masks
    didx = didx + 1

h5file.close()

# ######################## ends here ############################

# todo: Study the standard deviation and distribution of the dataset

# todo: Resize Scan in all dimensions

"""
