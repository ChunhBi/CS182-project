import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from skimage.metrics import structural_similarity as skssim
import time

img_dir = "pic/DIV2K_valid_HR/"

def psnr(img1, img2):
   mse = np.mean((img1/1.0 - img2/1.0) ** 2 )
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(255.0**2/mse)

def ssim(img1,img2):
   return skssim(img1,img2,multichannel=True)

img_names = os.listdir(img_dir)
images = []# the raw images
for name in img_names:
   img_path = os.path.join(img_dir, name)
   image = cv2.imread(img_path,)
   # b,g,r = cv2.split(image)
   # image = cv2.merge([r,g,b])
   images.append(image)

psnrList = [0,0,0,0,0]
ssimList = [0,0,0,0,0]
timeList = [0,0,0,0,0]
for groundTruth in images:
   print("Finish one epoch")
   groundTruth = cv2.resize(groundTruth,[groundTruth.shape[1]-groundTruth.shape[1]%4,groundTruth.shape[0]-groundTruth.shape[0]%4])
   img = groundTruth

   img = cv2.resize(groundTruth,dsize=None,fx=0.25,fy=0.25)
   # img = cv2.pyrDown(img)
   # img = cv2.pyrDown(img)

   t1 = time.time()
   resized = cv2.resize(img,dsize=None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
   t2 = time.time()
   psnrList[0] += psnr(groundTruth,resized)
   ssimList[0] += ssim(groundTruth,resized)
   timeList[0] += (t2-t1)

   sr = cv2.dnn_superres.DnnSuperResImpl_create()
   path = "model/EDSR_x4.pb"
   sr.readModel(path)
   sr.setModel("edsr",4)
   t1 = time.time()
   result = sr.upsample(img)
   t2 = time.time()
   psnrList[1] += psnr(groundTruth,result)
   ssimList[1] += ssim(groundTruth,result)
   timeList[1] += (t2-t1)


   sr = cv2.dnn_superres.DnnSuperResImpl_create()
   path = "model/ESPCN_x4.pb"
   sr.readModel(path)
   sr.setModel("espcn",4)
   t1 = time.time()
   result = sr.upsample(img)
   t2 = time.time()
   psnrList[2] += psnr(groundTruth,result)
   ssimList[2] += ssim(groundTruth,result)
   timeList[2] += (t2-t1)

   sr = cv2.dnn_superres.DnnSuperResImpl_create()
   path = "model/FSRCNN_x4.pb"
   sr.readModel(path)
   sr.setModel("fsrcnn",4)
   t1 = time.time()
   result = sr.upsample(img)
   t2 = time.time()
   psnrList[3] += psnr(groundTruth,result)
   ssimList[3] += ssim(groundTruth,result)
   timeList[3] += (t2-t1)

   sr = cv2.dnn_superres.DnnSuperResImpl_create()
   path = "model/LapSRN_x4.pb"
   sr.readModel(path)
   sr.setModel("lapsrn",4)
   t1 = time.time()
   result = sr.upsample(img)
   t2 = time.time()
   psnrList[4] += psnr(groundTruth,result)
   ssimList[4] += ssim(groundTruth,result)
   timeList[4] += (t2-t1)


psnrList = [i/len(images) for i in psnrList]
ssimList = [i/len(images) for i in ssimList]
timeList = [i/len(images) for i in timeList]

print("Bicubic PSNR:",psnrList[0])
print("SSIM:",ssimList[0])
print("Time spent:",timeList[0],"\n")

print("EDSR PSNR:",psnrList[1])
print("SSIM:",ssimList[1])
print("Time spent:",timeList[1],"\n")

print("ESPCN PSNR:",psnrList[2])
print("SSIM:",ssimList[2])
print("Time spent:",timeList[2],"\n")

print("FSRCNN PSNR:",psnrList[3])
print("SSIM:",ssimList[3])
print("Time spent:",timeList[3],"\n")

print("LapSRN PSNR:",psnrList[4])
print("SSIM:",ssimList[4])
print("Time spent:",timeList[4],"\n")