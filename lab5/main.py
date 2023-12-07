import cv2
import numpy as np
import os

image = cv2.imread('euro.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
cv2.imwrite("gray.jpeg", gray)
cv2.imwrite("blurred.jpeg", blurred)
cv2.imwrite("canny.jpeg", edges)


contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_area = 500000

output_dir = 'contour_images'
os.makedirs(output_dir, exist_ok=True)

count_500 = 0
count_20 =0

for i, contour in enumerate(contours):
    if cv2.contourArea(contour) < min_area:
        continue
    x, y, w, h = cv2.boundingRect(contour)
    cropped_region = image[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(output_dir, f'contour_{w}.jpg'), cropped_region)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if 1200>w and w >= 1100:
        count_500 += 1
    elif 1000>w and w >= 900:
        count_20 += 1


print("500 euros:", count_500)
print("20 euros:", count_20)
print("Total: ", count_500*500 + count_20*20, "euros")

