import cv2
import numpy as np
import matplotlib.pyplot as plt


def cut_graph(img,r,c,rSize,cSize):
    return img[r-rSize:r+rSize,c-cSize:c+cSize,:]

# input_path = "pic/test/0803.png"
# img = cv2.imread(input_path)
# img = cut_graph(img,img.shape[0]//2,img.shape[1]//2,200,200)
# cv2.imshow("img",img)
# cv2.waitKey()