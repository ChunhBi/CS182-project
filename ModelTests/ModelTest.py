import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from skimage.metrics import structural_similarity as skssim

input_path = "pic/test/0803.png"

def psnr(img1, img2):
   mse = np.mean((img1/1.0 - img2/1.0) ** 2 )
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(255.0**2/mse)

def ssim(img1,img2):
   return skssim(img1,img2,multichannel=True)

# Read image
groundTruth = cv2.imread(input_path)
groundTruth = cv2.resize(groundTruth,[groundTruth.shape[1]-groundTruth.shape[1]%4,groundTruth.shape[0]-groundTruth.shape[0]%4])
img = groundTruth
img = cv2.resize(groundTruth,dsize=None,fx=0.25,fy=0.25)
# img2 = cv2.pyrDown(img)
# img2 = cv2.pyrDown(img)

# print("change format")
# b,g,r = cv2.split(img)
# tmp_img = cv2.merge([r,g,b])
# b,g,r = cv2.split(img2)
# tmp_img2 = cv2.merge([r,g,b])

# plt.subplot(1,2,1)
# plt.imshow(tmp_img)
# plt.subplot(1,2,2)
# plt.imshow(tmp_img2)
# plt.show()




# Resized image
resized = cv2.resize(img,dsize=None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
print("Finished opencv")

cv2.imwrite("output/origin.png", groundTruth)
print("Origin PSNR:",psnr(groundTruth,groundTruth))
print("Origin SSIM:",ssim(groundTruth,groundTruth))
cv2.imwrite("output/bicubic.png", resized)
print("Bicubic PSNR:",psnr(groundTruth,resized))
print("Bicubic SSIM:",ssim(groundTruth,resized))


sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "model/EDSR_x4.pb"
sr.readModel(path)
sr.setModel("edsr",4)
t1 = time.time()
result = sr.upsample(img)
t2 = time.time()
cv2.imwrite("output/EDSR.png", result)
print("Finished EDSR")
print("EDSR PSNR:",psnr(groundTruth,result))
print("EDSR SSIM:",ssim(groundTruth,result))
print("EDSR time spent:",t2-t1)


sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "model/ESPCN_x4.pb"
sr.readModel(path)
sr.setModel("espcn",4)
t1 = time.time()
result = sr.upsample(img)
t2 = time.time()
cv2.imwrite("output/ESPCN.png", result)
print("Finished ESPCN")
print("ESPCN PSNR:",psnr(groundTruth,result))
print("ESPCN SSIM:",ssim(groundTruth,result))
print("ESPCN time spent:",t2-t1)


sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "model/FSRCNN_x4.pb"
sr.readModel(path)
sr.setModel("fsrcnn",4)
t1 = time.time()
result = sr.upsample(img)
t2 = time.time()
cv2.imwrite("output/FSRCNN.png", result)
print("Finished FSRCNN")
print("FSRCNN PSNR:",psnr(groundTruth,result))
print("FSRCNN SSIM:",ssim(groundTruth,result))
print("FSRCNN time spent:",t2-t1)


sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "model/LapSRN_x4.pb"
sr.readModel(path)
sr.setModel("lapsrn",4)
t1 = time.time()
result = sr.upsample(img)
t2 = time.time()
cv2.imwrite("output/LapSRN.png", result)
print("Finished LapSRN")
print("LapSRN PSNR:",psnr(groundTruth,result))
print("LapSRN SSIM:",ssim(groundTruth,result))
print("LapSRN time spent:",t2-t1)