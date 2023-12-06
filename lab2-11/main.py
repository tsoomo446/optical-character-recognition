import cv2
import numpy as np

# Load the input image
image = cv2.imread("../Images/motherboard.tif", cv2.IMREAD_GRAYSCALE)
def custom_avg(image, kernel):
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # padding олно. Мөн Zero-padding арга хэрэглэнэ.      
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    output_image = np.zeros_like(image)
    for y in range(pad_height, img_height - pad_height):
        for x in range(pad_width, img_width - pad_width):
            # Хөрш пикселүүдийн утгыг олоод кернелээрээ үржүүлнэ.
            neighborhood = image[y - pad_height:y + pad_height + 1, x - pad_width:x + pad_width + 1]
            result = np.sum(neighborhood * kernel)
            output_image[y, x] = result

def custom_median(image, kernel):
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # padding олно. Мөн Zero-padding арга хэрэглэнэ.      
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    output_image = np.zeros_like(image)
    for y in range(pad_height, img_height - pad_height):
        for x in range(pad_width, img_width - pad_width):
            # Хөрш пикселүүдийн утгыг олоод кернелээрээ үржүүлнэ.
            neighborhood = image[y - pad_height:y + pad_height + 1, x - pad_width:x + pad_width + 1]
            output_image[y, x] = np.median(neighborhood)

    return output_image

size = 3

results = [image]
results.append(image)

kernel = np.ones((size, size), np.float32) / (size * size)

# Задгай
custom_median_img = custom_median(image, kernel)
custom_avg_img = custom_avg(image, kernel)

results.append(custom_avg_img)
results.append(custom_median_img)


result = cv2.hconcat(results)

cv2.imwrite("cv_result.png", result)
