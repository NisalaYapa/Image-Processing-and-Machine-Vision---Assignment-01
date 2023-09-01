import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the image in LAB color space
image_lab = cv.cvtColor(cv.imread('highlights_and_shadows.jpg'), cv.COLOR_BGR2Lab)

# Extract the L channel
L_channel = image_lab[:, :, 0]

# Apply gamma correction
gamma = 1.8  # You can adjust this value
L_corrected = np.power(L_channel / 255.0, 1.0 / gamma) * 255.0

# Convert back to 8-bit range
L_corrected = np.clip(L_corrected, 0, 255).astype(np.uint8)

# Replace the L channel in the LAB image with the corrected values
image_lab[:, :, 0] = L_corrected

# Convert back to RGB color space
image_rgb_corrected = cv.cvtColor(image_lab, cv.COLOR_Lab2RGB)

# Display original and corrected images with color
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(cv.imread('highlights_and_shadows.jpg'), cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_rgb_corrected)
plt.title('Corrected Image')
plt.axis('off')

plt.tight_layout()
plt.show()

level = ('L', 'a', 'b')
colors1 = ('b', 'g', 'r')
colors2 = ('g', 'r', 'b')

plt.figure(figsize=(5, 15))  # Adjust the figure size for vertical histograms

for i, c in enumerate(level):
    plt.subplot(3, 1, i + 1)  # 3 rows, 1 column of subplots, index starting from 1
    hist_orig = cv.calcHist([cv.cvtColor(cv.imread('highlights_and_shadows.jpg'), cv.COLOR_BGR2Lab)], [i], None, [256], [0, 256])
    plt.plot(hist_orig, color=colors1[i], label='Original')

    hist_gamma = cv.calcHist([image_lab], [i], None, [256], [0, 256])  # Use image_lab here
    plt.plot(hist_gamma, color=colors2[i], label='Gamma')

    plt.title(f'Channel {c} Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()  # Arrange subplots neatly
plt.show()
