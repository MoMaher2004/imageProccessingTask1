import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image = cv2.imread('OIP.jfif')

# Convert BGR to RGB
image_rgb = image

# Plot histograms
plt.figure(figsize=(10, 5))

# Compute histograms for each channel
hist_b = cv2.calcHist([image_rgb], [0], None, [256], [0,256])
hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0,256])
hist_r = cv2.calcHist([image_rgb], [2], None, [256], [0,256])

# Plot original image
cv2.imshow("original",image_rgb)

# plot original hist
plt.subplot(3, 3, 1)
plt.bar(np.arange(256), hist_r[:,0], color='red')
plt.title('original image Red Channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(3, 3, 2)
plt.bar(np.arange(256), hist_g[:,0], color='green')
plt.title('original image Green Channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(3, 3, 3)
plt.bar(np.arange(256), hist_b[:,0], color='blue')
plt.title('original image Blue Channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# plt.tight_layout()
# plt.show()





# decrease brightness
alpha = 1  # Contrast control (1.0-3.0)
beta = -100  # Brightness control (-100 to 100)
adjusted_image = cv2.convertScaleAbs(image_rgb,alpha=alpha,beta=beta)

# Plot new image
cv2.imshow("less brightness only",adjusted_image)
hist_b = cv2.calcHist([adjusted_image], [0], None, [256], [0,256])
hist_g = cv2.calcHist([adjusted_image], [1], None, [256], [0,256])
hist_r = cv2.calcHist([adjusted_image], [2], None, [256], [0,256])

# plot new hist
plt.subplot(3, 3, 4)
plt.bar(np.arange(256), hist_r[:,0], color='red')
plt.title('less brightness Red Channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(3, 3, 5)
plt.bar(np.arange(256), hist_g[:,0], color='green')
plt.title('less brightness Green Channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(3, 3, 6)
plt.bar(np.arange(256), hist_b[:,0], color='blue')
plt.title('less brightness Blue Channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')




# Plot new image
# increase contrast decrease brightness
alpha = 1.5  # Contrast control (1.0-3.0)
beta = -100  # Brightness control (-100 to 100)
adjusted_image2 = cv2.convertScaleAbs(image_rgb,alpha=alpha,beta=beta)
cv2.imshow("less brightness high contrast",adjusted_image2)
hist_b = cv2.calcHist([adjusted_image2], [0], None, [256], [0,256])
hist_g = cv2.calcHist([adjusted_image2], [1], None, [256], [0,256])
hist_r = cv2.calcHist([adjusted_image2], [2], None, [256], [0,256])

# plot new hist
plt.subplot(3, 3, 7)
plt.bar(np.arange(256), hist_r[:,0], color='red')
plt.title('less brightness high contrast Red Channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(3, 3, 8)
plt.bar(np.arange(256), hist_g[:,0], color='green')
plt.title('less brightness high contrast Green Channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(3, 3, 9)
plt.bar(np.arange(256), hist_b[:,0], color='blue')
plt.title('less brightness high contrast Blue Channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
