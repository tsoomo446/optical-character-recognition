import numpy as np
import cv2

image = cv2.imread("../Images/rose.tif")

# Nearest interpolation ашиглан 4 дахин томруулах хэсэг
scale_factor = 4    
height, width, channels = image.shape

new_width = width * scale_factor
new_height = height * scale_factor

upscaled_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

for y in range(new_height):
    for x in range(new_width):
        # Хамгийн ойр утгатай кординатыг олох
        src_x = int(x / scale_factor)
        src_y = int(y / scale_factor)
        upscaled_image[y, x] = image[src_y, src_x]

# OpenCV2 - ын resize функцыг ашиглан томруулах хэсэг
dim = (new_width, new_height)

function_linear = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
function_nearest = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)

# Задгайгаар бичсэн зурагнаас хэр ялгаатай байгаа эсэхийг харах 
sub_linear = function_linear - upscaled_image
sub_nearest = function_nearest - upscaled_image

# Томруулсан зургаа буцааж жижигсгээд алдааг нь харах хэсэг
function_linear_error = image - cv2.resize(function_linear, (height, width), interpolation=cv2.INTER_LINEAR)
function_nearest_error = image - cv2.resize(function_nearest, (height, width), interpolation=cv2.INTER_NEAREST)

result = cv2.hconcat([function_linear, function_nearest, sub_linear, sub_nearest])
error_result = cv2.hconcat([function_linear_error, function_nearest_error])

# Зурагнуудаа хадгалах
cv2.imwrite("upscaled.png", upscaled_image)
cv2.imwrite("result.png",result)
cv2.imwrite("error_result.png", error_result)