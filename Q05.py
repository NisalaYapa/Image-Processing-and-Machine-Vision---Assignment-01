import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization_custom(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate histogram
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])

    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Apply histogram equalization
    equalized_image = np.interp(image.flatten(), range(256), cdf_normalized).reshape(image.shape)

    # Calculate histograms
    original_hist = hist
    equalized_hist ,__= np.histogram(equalized_image.flatten(), bins=256, range=[0, 256])

    # Display histograms
    plt.figure(figsize=(10, 6))

    plt.subplot(221)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(222)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')

    plt.subplot(223)
    plt.plot(original_hist, color='blue')
    plt.title('Original Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.subplot(224)
    plt.plot(equalized_hist, color='orange')
    plt.title('Equalized Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Call the function with the path to your image
image_path = 'shells.tif'
histogram_equalization_custom(image_path)
