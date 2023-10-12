import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('../Images/coffee.tif', cv2.IMREAD_GRAYSCALE)
minmax_func = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

minmax_img = np.zeros((image.shape[0],image.shape[1]),dtype = 'uint8')
binary_img = np.zeros((image.shape[0],image.shape[1]),dtype = 'uint8')


for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        minmax_img[i,j] = 255*((image[i,j]-np.min(image))/(np.max(image)-np.min(image)))
        if minmax_img[i,j] > 150:
            binary_img[i,j] = 255 
        else:
            binary_img[i,j] = 0
 

sub = minmax_img - minmax_func
result = cv2.hconcat([image, minmax_img, binary_img, minmax_func, sub])
cv2.imwrite("result.png", result)

# Histogram харьцуулалт
hist1 = cv2.calcHist([image], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([minmax_img], [0], None, [256], [0, 256])
plt.figure(figsize=(10, 6))
plt.plot(hist1, color='blue', label='Анхны зураг')
plt.plot(hist2, color='red', label='MinMax stretch хийсэн зураг')
plt.title('Histogram харьцуулалт')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.grid()
plt.show()