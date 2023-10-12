import cv2
import numpy as np

# skull.tif зургаа GRAYSCALE төрлөөр унших
img = cv2.imread("../Images/skull.tif", cv2.IMREAD_GRAYSCALE)

# 2-7bit ялгаралын түвшинтэй зурагнуудаа үүсгэх
bit7 = (img // 2) * 2
bit6 = (img // 4) * 4
bit5 = (img // 8) * 8
bit4 = (img // 16) * 16
bit3 = (img // 32) * 32
bit2 = (img // 64) * 64

bit1 = np.zeros_like(img, dtype=np.uint8)

# 1bit ын буюу binary зурагаа 100 утгаар threshold хийн үүсгэх
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i, j] > 100:
            bit1[i, j] = 255
        else:
            bit1[i, j] = 0

# Үр дүнгээ нэгтгэн result гэсэн зураг үүсгээд result.png file-д хадгалах
column1 = cv2.hconcat([img, bit7, bit6, bit5])
column2 = cv2.hconcat([bit4, bit3, bit2, bit1])

result = cv2.vconcat([column1, column2])

cv2.imwrite("result.png", result)