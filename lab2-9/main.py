import cv2
import numpy as np

# Load the input image
image = cv2.imread("../Images/a.tif", cv2.IMREAD_GRAYSCALE)

def custom_filter2D(image, kernel):
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

    return output_image

kernel_sizes = [3, 5, 9,15,35]

results = [image]
cv2_result = [image]
for size in kernel_sizes:
    kernel = np.ones((size, size), np.float32) / (size * size)

    # Задгай
    custom_filtered_image = custom_filter2D(image, kernel)
    # CV2 filter2D функц
    filtered_image = cv2.filter2D(image, cv2.CV_8U, kernel)
    results.append(custom_filtered_image)
    cv2_result.append(filtered_image)

row1 = cv2.hconcat([results[0], results[1]])
row2 = cv2.hconcat([results[2], results[3]])
row3 = cv2.hconcat([results[4], results[5]])


result = cv2.vconcat([row1,row2,row3])
cv2.imwrite("result.png", result)

cv_row1 = cv2.hconcat([cv2_result[0], cv2_result[1]])
cv_row2 = cv2.hconcat([cv2_result[2], cv2_result[3]])
cv_row3 = cv2.hconcat([cv2_result[4], cv2_result[5]])


result = cv2.vconcat([cv_row1,cv_row2,cv_row3])
cv2.imwrite("cv_result.png", result)
