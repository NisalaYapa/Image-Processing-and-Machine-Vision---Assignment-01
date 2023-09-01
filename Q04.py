import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

# Load the image
image_path = 'spider.png'
image = cv.imread(image_path)
image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Split the image into hue, saturation, and value planes
hue, saturation, value = cv.split(image)

# Define the intensity transformation function
def intensity_transformation(x, a, sigma):
    return min(255, max(0, round(x + a * 128 * math.exp(-(x - 128)**2 / (2 * sigma**2)))))

# Apply the intensity transformation to the saturation plane
a = 0.9  # You can adjust this value for a visually pleasing output
sigma = 70
enhanced_saturation = np.vectorize(intensity_transformation)(saturation, a, sigma)

# Recombine the three planes
enhanced_image = cv.merge((hue, enhanced_saturation.astype(np.uint8), value))
enhanced_image = cv.cvtColor(enhanced_image, cv.COLOR_HSV2BGR)

# Display the original image, vibrance-enhanced image, and intensity transformation
plt.figure(figsize=(10, 10))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv.cvtColor(image, cv.COLOR_HSV2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f'Enhanced Image (a = {a})')
plt.imshow(cv.cvtColor(enhanced_image, cv.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
x_values = np.arange(256)
transformed_values = [intensity_transformation(x, a, sigma) for x in x_values]
plt.plot(x_values, transformed_values)
plt.title('Intensity Transformation')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')
plt.grid()

plt.tight_layout()
plt.show()
