import cv2
import numpy as np
import random
import os

def cut_256(img):
    r = random.randint(0, img.shape[0] - 256)
    c = random.randint(0, img.shape[1] - 256)
    return img[r:r+256, c:c+256,:]

## dir
# input_dir = "negative image set/"
# output_dir = "air_256/"
input_dir = "DIV2K_valid_HR/"
output_dir = "DIV2K_256/"

## cut input pic
input_names = os.listdir(input_dir)
for name in input_names:
   img_path = os.path.join(input_dir, name)
   image = cv2.imread(img_path,)
   # b,g,r = cv2.split(image)
   # image = cv2.merge([r,g,b])
   image = cut_256(image)
   cv2.imwrite(os.path.join(output_dir, name), image)
