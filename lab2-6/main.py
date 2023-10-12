import cv2
from matplotlib import pyplot as plt
import numpy as np

img1 = cv2.imread("../Images/coffee-1.tif", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("../Images/coffee-2.tif", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("../Images/coffee-3.tif", cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread("../Images/coffee-4.tif", cv2.IMREAD_GRAYSCALE)

cv2.imwrite("first.png", cv2.hconcat([img1, img2, img3, img4]))




eq1 = cv2.equalizeHist(img1)
eq2 = cv2.equalizeHist(img2)
eq3 = cv2.equalizeHist(img3)
eq4 = cv2.equalizeHist(img4)

# Histogram харьцуулалт
hist1 = cv2.calcHist([eq1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([eq2], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([eq3], [0], None, [256], [0, 256])
hist4 = cv2.calcHist([eq4], [0], None, [256], [0, 256])

cv2.imwrite("hist_eq.png", cv2.hconcat([eq1, eq2, eq3, eq4]))

plt.figure(figsize=(10, 6))
plt.plot(hist1, color='blue', label='1-р зураг')
plt.plot(hist2, color='red', label='2-р зураг')
plt.plot(hist3, color='green', label='3-р зураг')
plt.plot(hist4, color='yellow', label='4-р зураг')


plt.title('Histogram харьцуулалт')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.grid()
plt.show()