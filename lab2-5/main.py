import cv2
from matplotlib import pyplot as plt
import numpy as np

# Зургаа унших
image = cv2.imread('../Images/dollar.tif', cv2.IMREAD_GRAYSCALE)

bit_planes = []

for bit_index in range(8): 
    # bit_index-ээр shift хийх
    bitmask = 1 << bit_index

    bit_plane = (image & bitmask) * 255

    bit_planes.append(bit_plane)


row1 = cv2.hconcat([image, bit_planes[0], bit_planes[1]])
row2 = cv2.hconcat([bit_planes[2], bit_planes[3], bit_planes[4]])
row3 = cv2.hconcat([bit_planes[5], bit_planes[6], bit_planes[7]])

result = cv2.vconcat([row1, row2, row3])
cv2.imwrite("result.png", result)

