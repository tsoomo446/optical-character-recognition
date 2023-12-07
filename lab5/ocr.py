import cv2
import numpy as np
import pytesseract
import json

image = cv2.imread("contour_images/contour_1130.jpg", cv2.IMREAD_GRAYSCALE)

mask = np.zeros_like(image, dtype=np.uint8)


mask[ 340:500, 860:1120] = 255


masks = [mask]
mask_names = ["value"]
data = {}

for mask, name in zip(masks, mask_names):
    x, y, w, h = cv2.boundingRect(mask)
    cropped_region = image[y:y+h, x:x+w]
    cropped_region = cv2.GaussianBlur(cropped_region, (3, 3), 0)
    cropped_region = cv2.threshold(cropped_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite(f"data/{name}_cropped.png", cropped_region)
    text = pytesseract.image_to_string(cropped_region)
    print(f"{name} Text:")
    print(text)
    data[name] = text

with open("extracted_data.json", "w") as json_file:
    json.dump(data, json_file, indent=4)

print("Data saved in 'extracted_data.json'")
