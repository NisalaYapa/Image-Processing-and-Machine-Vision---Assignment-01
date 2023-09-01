import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

# Load the image
image = cv.imread('einstein.png', cv.IMREAD_GRAYSCALE)
kernal = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

# Sobel filter method one
image_q1 = cv.filter2D(image, -1, kernal)

# Sobel filter method two
def convolution(image1, kernal1, x, y):
    value = 0
    for i in range(3):
        for j in range(3):
            value += kernal1[i][j] * image1[x + 1 - i][y + 1 - j]
    return value

shape1 = np.shape(image)
image_q2 = np.zeros((shape1[0], shape1[1]))
for i in range(1, shape1[0] - 1):
    for j in range(1, shape1[1] - 1):
        image_q2[i][j] = np.clip(convolution(image, kernal, i, j), 0, 255)

# Sobel filter method three
def convolution2(image1, kernalH, kernalV, x, y):
    value = 0
    for i in range(3):
        value += (kernalH[0] * image1[x + 1 - i][y + 1] + kernalH[1] * image1[x + 1 - i][y + 0] + kernalH[2] * image1[x + 1 - i][y - 1]) * kernalV[i]
    return value

kernal1H = np.array([1, 0, -1])
kernal1V = np.array([1, 2, 1])

image_q3 = np.zeros((shape1[0], shape1[1]))
for i in range(1, shape1[0] - 1):
    for j in range(1, shape1[1] - 1):
        image_q3[i][j] = np.clip(convolution2(image, kernal1H, kernal1V, i, j), 0, 255)

# Display the three images horizontally with titles
plt.figure(figsize=(15, 5))

# Built-in Sobel
plt.subplot(131)
plt.imshow(image_q1, cmap='gray')
plt.title('Built-in Sobel')
plt.axis('off')

# Manual Sobel
plt.subplot(132)
plt.imshow(image_q2, cmap='gray')
plt.title('Manual Sobel')
plt.axis('off')

# Custom Sobel (method three)
plt.subplot(133)
plt.imshow(image_q3, cmap='gray')
plt.title('with matrix  property')
plt.axis('off')

plt.tight_layout()
plt.show()
