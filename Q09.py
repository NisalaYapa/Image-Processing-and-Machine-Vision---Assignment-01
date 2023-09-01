import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('flower.jpeg')

# Create a mask with the same size as the image
mask = np.zeros(image.shape[:2], np.uint8)

# Define the rectangle enclosing the object (flower)
rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)  # Adjust the coordinates as needed

# Initialize background and foreground models
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Apply GrabCut algorithm
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Modify the mask to create a binary segmentation result
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# (a) Show the final segmentation mask
plt.subplot(131), plt.imshow(mask2, cmap='gray')
plt.title('Segmentation Mask'), plt.xticks([]), plt.yticks([])

# (a) Show the foreground image
foreground = image * mask2[:, :, np.newaxis]
plt.subplot(132), plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
plt.title('Foreground Image'), plt.xticks([]), plt.yticks([])

# (a) Show the background image
background = image * (1 - mask2[:, :, np.newaxis])
plt.subplot(133), plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
plt.title('Background Image'), plt.xticks([]), plt.yticks([])

plt.show()

# (b) Produce an enhanced image with a substantially blurred background
# Apply Gaussian blur to the background
blurred_background = cv2.GaussianBlur(image * (1 - mask2[:, :, np.newaxis]), (0, 0), 4)  # Adjust the kernel size (15) as needed

# Create the enhanced image by combining the blurred background and foreground
enhanced_image = blurred_background + (image * mask2[:, :, np.newaxis])

# (b) Display the original image alongside the enhanced image
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
plt.title('Enhanced Image'), plt.xticks([]), plt.yticks([])

plt.show()

# Save the images to disk
cv2.imwrite('segmentation_mask.png', mask2 * 255)
cv2.imwrite('foreground_image.png', foreground)
cv2.imwrite('background_image.png', background)
cv2.imwrite('enhanced_image.png', enhanced_image)