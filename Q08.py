import cv2 as cv
import numpy as np

# Load the input image
image = cv.imread('zooming/im11small.png', cv.IMREAD_COLOR)
imageB, imageG, imageR = cv.split(image)

# Define the zoom factor
factor = 4

# Calculate the new dimensions
shape1 = image.shape
shape2 = (shape1[0] * factor, shape1[1] * factor)

# Create empty large images for each channel for both methods
large_imageB_nn = np.zeros((shape2[0], shape2[1]), dtype=np.uint8)
large_imageG_nn = np.zeros((shape2[0], shape2[1]), dtype=np.uint8)
large_imageR_nn = np.zeros((shape2[0], shape2[1]), dtype=np.uint8)

large_imageB_bi = np.zeros((shape2[0], shape2[1]), dtype=np.uint8)
large_imageG_bi = np.zeros((shape2[0], shape2[1]), dtype=np.uint8)
large_imageR_bi = np.zeros((shape2[0], shape2[1]), dtype=np.uint8)

# Perform nearest-neighbor interpolation for each channel
for i in range(shape2[0]):
    for j in range(shape2[1]):
        i_ = i // factor
        j_ = j // factor

        large_imageB_nn[i, j] = imageB[i_, j_]
        large_imageG_nn[i, j] = imageG[i_, j_]
        large_imageR_nn[i, j] = imageR[i_, j_]

# Perform bilinear interpolation for each channel
for i in range(shape2[0]):
    for j in range(shape2[1]):
        i_ = i / factor
        j_ = j / factor
        di = int(i_)
        dj = int(j_)

        if di < shape1[0] - 1 and dj < shape1[1] - 1:
            dx = i_ - di
            dy = j_ - dj

            pixel_A = imageB[di, dj]
            pixel_B = imageB[di, dj + 1]
            pixel_C = imageB[di + 1, dj]
            pixel_D = imageB[di + 1, dj + 1]

            large_imageB_bi[i, j] = (1 - dx) * (1 - dy) * pixel_A + dx * (1 - dy) * pixel_B + (1 - dx) * dy * pixel_C + dx * dy * pixel_D

            pixel_A = imageG[di, dj]
            pixel_B = imageG[di, dj + 1]
            pixel_C = imageG[di + 1, dj]
            pixel_D = imageG[di + 1, dj + 1]

            large_imageG_bi[i, j] = (1 - dx) * (1 - dy) * pixel_A + dx * (1 - dy) * pixel_B + (1 - dx) * dy * pixel_C + dx * dy * pixel_D

            pixel_A = imageR[di, dj]
            pixel_B = imageR[di, dj + 1]
            pixel_C = imageR[di + 1, dj]
            pixel_D = imageR[di + 1, dj + 1]

            large_imageR_bi[i, j] = (1 - dx) * (1 - dy) * pixel_A + dx * (1 - dy) * pixel_B + (1 - dx) * dy * pixel_C + dx * dy * pixel_D

# Merge the channels back into a single color image for both methods
large_merge_nn = cv.merge((large_imageB_nn, large_imageG_nn, large_imageR_nn))
large_merge_bi = cv.merge((large_imageB_bi, large_imageG_bi, large_imageR_bi))

# Display the zoomed-in images for both methods
cv.imshow('Nearest Neighbor Interpolation', large_merge_nn)
cv.imshow('Bilinear Interpolation', large_merge_bi)
cv.waitKey(0)
cv.destroyAllWindows()
