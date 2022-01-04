import cv2
from utils import *
import numpy as np

input_path = "output/"

img = cv2.imread(input_path + "origin.png")
r = img.shape[0]//2 - 450
c = img.shape[1]//2 + 350
rSize = 200
cSize = 200


img = cv2.imread(input_path + "origin.png")
img = cut_graph(img,r,c,rSize,cSize)
cv2.imwrite("output/cut_origin.png", img)

img = cv2.imread(input_path + "bicubic.png")
img = cut_graph(img,r,c,rSize,cSize)
cv2.imwrite("output/cut_bicubic.png", img)

img = cv2.imread(input_path + "EDSR.png")
img = cut_graph(img,r,c,rSize,cSize)
cv2.imwrite("output/cut_EDSR.png", img)

img = cv2.imread(input_path + "ESPCN.png")
img = cut_graph(img,r,c,rSize,cSize)
cv2.imwrite("output/cut_ESPCN.png", img)

img = cv2.imread(input_path + "FSRCNN.png")
img = cut_graph(img,r,c,rSize,cSize)
cv2.imwrite("output/cut_FSRCNN.png", img)

img = cv2.imread(input_path + "LapSRN.png")
img = cut_graph(img,r,c,rSize,cSize)
cv2.imwrite("output/cut_LapSRN.png", img)
