import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Create the transformation array
t1 = np.linspace(0, 0, 86).astype('uint8')
t2 = np.linspace(50, 50, 95).astype('uint8')
t3 = np.linspace(180, 180, 30).astype('uint8')
t4 = np.linspace(255, 255, 45).astype('uint8')

transform = np.concatenate((t1, t2), axis=0).astype('uint8')
transform = np.concatenate((transform, t3), axis=0).astype('uint8')
transform = np.concatenate((transform, t4), axis=0).astype('uint8')

# Create the transformation plot
fig, ax = plt.subplots()
ax.plot(transform)
ax.set_xlabel(r'Input, $f(\mathbf{x})$')
ax.set_ylabel('Output, $\mathrm{T}[f(\mathbf{x})]$')
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
ax.set_aspect('equal')
plt.savefig('transform.png')
plt.show()

# Load the original image
img_orig = cv.imread(r'BrainProtonDensitySlice9.png', cv.IMREAD_GRAYSCALE)

# Apply the transformation
image_transformed = cv.LUT(img_orig, transform)

# Apply median filtering
filtered_image = cv.medianBlur(image_transformed, 3)  # 3x3 neighborhood

# Create a figure with three subplots in a horizontal row
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(131)
plt.imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB), cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Transformed Image
plt.subplot(132)
plt.imshow(image_transformed, cmap='gray')
plt.title('Transformed Image')
plt.axis('off')

# Filtered Image
plt.subplot(133)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')

# Display the plot
plt.tight_layout()
plt.show()

# Wait for a key press and then close the OpenCV window
cv.waitKey(0)
cv.destroyAllWindows()
