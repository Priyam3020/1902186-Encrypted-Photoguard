import numpy as np
import cv2
import matplotlib.pyplot as plt
matplotlib inline
image = cv2.imread('index.png')
rows,cols = image.shape[:2]
col/2,rows/2) is the center of rotation for the image
 M is the cordinates of the center
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv2.warpAffine(image,M,(cols,rows))
plt.imshow(dst)
for m,n in matches:
if m.distance < 0.75*n.distance:
good.append([m])
image3 = cv2.drawMatchesKnn(image1,kp1,image2,kp2,good,flags = 2)