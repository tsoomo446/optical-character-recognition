import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('../Images/skeleton.tif', cv2.IMREAD_GRAYSCALE)
laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
laplacian_result = cv2.filter2D(image, -1, laplacian_kernel)
sharpened = cv2.add(image, laplacian_result)
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_result = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
sobel_avg = cv2.blur(sobel_result, (5, 5))
sub = cv2.subtract(sharpened, sobel_avg)
sharpened_2 = cv2.add(image, sub)
gamma = 4.0
last_result = cv2.pow(sharpened_2 / 255., gamma) * 255
last_result = np.clip(sharpened_2, 0, 255).astype(np.uint8)
r1 = cv2.hconcat([image, laplacian_result,  sobel_avg, sub])
r2 = cv2.hconcat([sharpened, sobel_result, sharpened_2, last_result])
result = cv2.vconcat([r1,r2])
cv2.imwrite('result.png', result)
comp1 = cv2.subtract(image, last_result)
comp2 = cv2.subtract(image, sharpened_2)
comp_result = cv2.hconcat([comp1, comp2])
cv2.imwrite('comp.png', comp_result)