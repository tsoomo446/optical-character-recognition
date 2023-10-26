import cv2
from matplotlib import pyplot as plt
import numpy as np

image = cv2.imread("../Images/enhacement.tif", cv2.IMREAD_GRAYSCALE)

grid_size = (3,3)
image_height, image_width = image.shape

#local_enhancement тухайн region хэсэг дээр хийх функц
def local_enhancement(region):
    enhanced_region = cv2.equalizeHist(region)
    return enhanced_region

output_image_3x3 = np.zeros_like(image)
# 3x3 kernel бүрийн хувьд local_enhancement хийнэ
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        start_x = (i * image_height) // grid_size[0]
        end_x = ((i + 1) * image_height) // grid_size[0]
        start_y = (j * image_width) // grid_size[1]
        end_y = ((j + 1) * image_width) // grid_size[1]

        region = image[start_x:end_x, start_y:end_y]

        enhanced_region = local_enhancement(region)

        output_image_3x3[start_x:end_x, start_y:end_y] = enhanced_region

img_function = cv2.equalizeHist(image)


result = cv2.hconcat([image, img_function, output_image_3x3])
cv2.imwrite("result.png", result)

# Histogram харьцуулалт
hist1 = cv2.calcHist([image], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([img_function], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([output_image_3x3], [0], None, [256], [0, 256])

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