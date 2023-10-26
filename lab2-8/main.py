import cv2
import numpy as np

image = cv2.imread("../Images/einstein.tif", cv2.IMREAD_GRAYSCALE)

# and mask - аа үүсгээд 255 тодорхой сегментийг цагаан болгоно
and_mask = np.zeros_like(image, dtype=np.uint8)
and_mask[10:300,250:490] = 255

# and image үүсгэнэ мөн задгайгаар & оператор ашиглаж болно.
and_image = cv2.bitwise_and(image, and_mask)
and_image_custom = image & and_mask

# or mask - аа үүсгээд 255 тодорхой сегментийг хар болгоно
or_mask = np.full(image.shape, 255, dtype=np.uint8)
or_mask[10:300,250:490] = 0

# or image үүсгэнэ мөн задгайгаар | оператор ашиглаж болно.
or_image = cv2.bitwise_or(image, or_mask)
or_image_custom = image | or_mask

row1 = cv2.hconcat([image, and_mask, and_image, and_image_custom])
row2 = cv2.hconcat([image, or_mask, or_image, or_image_custom])

result = cv2.vconcat([row1,row2])

cv2.imwrite("result.png", result)