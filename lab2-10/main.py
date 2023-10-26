import cv2
from matplotlib import pyplot as plt
import numpy as np

# Load the input image
image = cv2.imread("../Images/galaxy.tif", cv2.IMREAD_GRAYSCALE)

results = [image]
kernel = np.ones((15, 15), np.float32) / (15 * 15)
# Зургаа 15X15 маскаар averaging filter хийнэ
filtered_image = cv2.filter2D(image, -1, kernel)
results.append(filtered_image)
threshold = np.zeros_like(image, dtype=np.uint8)

for i in range(filtered_image.shape[0]):
    for j in range(filtered_image.shape[1]):
        # 65 утгаар threshold хийнэ
        if filtered_image[i, j] > 65:
            threshold[i, j] = 255
        else:
            threshold[i, j] = 0
results.append(threshold)
result = cv2.hconcat(results)
cv2.imwrite("result.png", result)

# Histogram
hist1 = cv2.calcHist([results[0]], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([results[1]], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([results[2]], [0], None, [256], [0, 256])

plt.figure(figsize=(10, 6))
plt.plot(hist1, color='blue', label='1-р зураг')
plt.plot(hist2, color='red', label='2-р зураг')
plt.plot(hist3, color='green', label='3-р зураг')


plt.title('Histogram харьцуулалт')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.grid()
plt.show()