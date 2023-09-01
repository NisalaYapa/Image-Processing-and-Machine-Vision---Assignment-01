import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

# Load the image
image_path = 'jeniffer.jpg'
image = cv.imread(image_path)
image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Split the image into hue, saturation, and value planes
hue, saturation, value = cv.split(image)

plt.subplot(2, 3, 1)
plt.title('Hue')
plt.imshow(hue, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Saturation')
plt.imshow(saturation, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Value')
plt.imshow(value, cmap='gray')
plt.axis('off')

selected_channel = saturation

threshold_value = 20  # Adjust this value based on your image and requirement
ret, foreground_mask = cv.threshold(selected_channel, threshold_value, 255, cv.THRESH_BINARY)


plt.subplot(2, 3, 4)
plt.title('Foreground Mask')
plt.imshow(foreground_mask, cmap='gray')
plt.axis('off')

foreground = cv.bitwise_and(selected_channel, selected_channel, mask=foreground_mask)
histogram = cv.calcHist([foreground], [0], None, [256], [0, 256])
cdf_before = np.cumsum(histogram)

plt.subplot(2, 3, 5)
plt.plot(cdf_before, color='b')
plt.title('Histogram before Equalization')
plt.xlabel('Pixel Value')
plt.ylabel('Cumulative Frequency')

equalized_values = cv.equalizeHist(foreground)
histogram2 = cv.calcHist([equalized_values], [0], None, [256], [0, 256])
cdf_after = np.cumsum(histogram2)
plt.subplot(2, 3, 6)
plt.plot(cdf_after, color='r')
plt.title('Histogram after Equalization')
plt.xlabel('Pixel Value')
plt.ylabel('Cumulative Frequency')

plt.tight_layout()
plt.show()

processed_hsv_image = cv.merge((hue, equalized_values, value))

# Convert the HSV image back to RGB
final_rgb_image = cv.cvtColor(processed_hsv_image, cv.COLOR_HSV2RGB)
original_image = cv.cvtColor(image, cv.COLOR_HSV2RGB)

plt.subplot(2, 1, 1)
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

# Display the final RGB image
plt.subplot(2, 1, 2)
plt.imshow(final_rgb_image)
plt.title('Final Image')
plt.axis('off')
plt.tight_layout()
plt.show()
