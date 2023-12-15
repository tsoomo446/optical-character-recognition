import cv2
import matplotlib.pyplot as plt

image = cv2.imread('../Images/dipxe.tif', cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
mask = cv2.subtract(image, blurred)
unsharp_image = cv2.add(image, mask)
highboost_image = cv2.add(image, 10 * mask)

# Үр дүн 
results = [image, blurred, mask, unsharp_image, highboost_image]
result = cv2.vconcat(results)
cv2.imwrite('result.png', result)

# Гистограм
hist1 = cv2.calcHist([results[0]], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([results[1]], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([results[3]], [0], None, [256], [0, 256])
hist4 = cv2.calcHist([results[4]], [0], None, [256], [0, 256])

plt.figure(figsize=(10, 6))
plt.plot(hist1, color='blue', label='1-р зураг')
plt.plot(hist2, color='red', label='2-р зураг')
plt.plot(hist3, color='green', label='3-р зураг')
plt.plot(hist4, color='black', label='4-р зураг')


plt.title('Histogram харьцуулалт')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.grid()
plt.show()