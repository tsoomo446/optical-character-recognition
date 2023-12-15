import cv2
import numpy as np

image = cv2.imread('../Images/moon.tif', cv2.IMREAD_GRAYSCALE)

scale_factor = 10.0
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
kernel_d = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])


laplacian_result = cv2.filter2D(image, -1, kernel)
scaled = scale_factor * laplacian_result
laplacian_result = np.uint8(np.absolute(laplacian_result))
scaled_result = np.uint8(np.absolute(scaled))

laplacian_c = cv2.filter2D(image, -1, kernel)
laplacian_d = cv2.filter2D(image, -1, kernel_d)
result_c = image + np.uint8(np.absolute(laplacian_c))   
result_d = image + np.uint8(np.absolute(laplacian_d))

result = cv2.hconcat([image,laplacian_result, scaled_result, result_c, result_d])

cv2.imwrite('result.png', result)