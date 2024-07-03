import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from PIL import Image
import cv2

elvis_image = cv2.imread("/Users/kasi/Downloads/elvis-1.bmp")

#Question 1a
columnMin = 150
columnMax = 200
row = 120
region = elvis_image[row, columnMin:columnMax, 0]
x = np.arange(columnMin, columnMax)

plt.grid(True)
plt.figure(figsize=(12, 6))
plt.stem(x, region, basefmt='b-')
plt.xlabel('Column')
plt.ylabel('Pixel Value for Channel 0')
plt.title('Pixel Values for Row 120 (Original image)')
plt.show()

#Question 1b 
kernel = np.array([[0, -0.25, -0],
                   [-0.25,  2, -0.25],
                   [0, -0.25, 0]])

convolved_image = cv2.filter2D(elvis_image, -1, kernel)

# Compare the original and convolved images
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(elvis_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(122)
plt.title('Convolved Image')
plt.imshow(cv2.cvtColor(convolved_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

#Question 1c
region = convolved_image[row, columnMin:columnMax, 0]
x = np.arange(columnMin, columnMax)
plt.grid(True)
plt.figure(figsize=(12, 6))
plt.stem(x, region, basefmt='c-')
plt.title('CVL Image')
plt.xlabel('Column')
plt.ylabel('Pixel Value for Channel 0')
plt.show()

#original
conv_gray = cv2.cvtColor(convolved_image, cv2.COLOR_BGR2GRAY) 
og_gray = cv2.cvtColor(elvis_image, cv2.COLOR_BGR2GRAY)
cvlhistogram = cv2.calcHist([conv_gray], [0], None, [256], [0, 256]) 
originalhistogram = cv2.calcHist([og_gray], [0], None, [256], [0, 256])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.grid(True) 
plt.plot(originalhistogram, color='b') 
plt.xlabel('Pixels')
plt.ylabel('Frequency')
plt.title('Original Image')
plt.show()

#cvl image
plt.grid(True) 
plt.plot(cvlhistogram, color='r')
plt.xlabel('Pixels') 
plt.ylabel('Frequency')
plt.title('CVL Image') 
plt.tight_layout()
plt.show()
