from matplotlib import pyplot as plt
import numpy as np
import cv2

image = cv2.imread("../Images/kidney.tif", cv2.IMREAD_GRAYSCALE)

row, col = image.shape

print(image[0,0])

j = np.zeros_like(image)
k = np.zeros_like(image)

for x in range(row):
    for y in range(col):
        if image[x, y] > 150:
            j[x, y] = 255
        else:
            j[x, y] = 0

for x in range(row):
    for y in range(col):
        if image[x, y] > 150:
            k[x, y] = 240
        elif 70 < image[x, y] and image[x,y] < 150: 
            k[x, y] = 0
        else:
            k[x, y] = 142

result = cv2.hconcat([image, j, k])
cv2.imwrite("result.png", result)


# Histogram харьцуулалт
hist1 = cv2.calcHist([image], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([k], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([j], [0], None, [256], [0, 256])
plt.figure(figsize=(10, 6))
plt.plot(hist1, color='blue', label='Анхны зураг')
plt.plot(hist2, color='red', label='3-level зураг')
plt.plot(hist3, color='green', label='Binary зураг')
plt.title('Histogram харьцуулалт')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.grid()
plt.show()
