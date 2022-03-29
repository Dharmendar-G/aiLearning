import os
import cv2
import numpy as np
from PIL import Image, ImageStat
import shutil

#path of main images folder.
image_folder = r'D:/YouTube/images'
#Path of duplicate images to move.
duplicate_dir = "D:/YouTube/DuplicateImages"
#Path of unique images to move.
unique_dir = "D:/YouTube/UniqueImages"


image_files = [_ for _ in os.listdir(image_folder) if _.endswith('jpg')]
list_of_mean = []
for img in image_files:
    image_org = Image.open(os.path.join(image_folder, img))
    pix_mean1 = ImageStat.Stat(image_org).mean
    mat1 = format((np.median(pix_mean1)),'.1f')
    list_of_mean.append(mat1)
    # print(mat1)

#Create dictionary having image name and its value
print("Length of images in folder: ",len(image_files))
print("Length of mean value in folder: ",len(list_of_mean))

#Dictionary
D = dict(zip(image_files, list_of_mean))
# print(D)

uni = []

for name, mean in D.items():
    # print(name, mean)
    if list_of_mean.count(mean) >=1:
        if mean not in uni:
            print("unique keys", name)
            shutil.copy(os.path.join(image_folder, name), unique_dir)
            print("unique values: ", mean)
            uni.append(mean)
        else:
            shutil.copy(os.path.join(image_folder, name), duplicate_dir)
print(uni)

