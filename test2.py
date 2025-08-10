import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('reflect_image.jpg')

# 1
image_1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
mask = cv2.inRange(image_1, np.array([0,0,180]), np.array([180,28,255]))                       
mask_white = cv2.bitwise_and(image_1, image_1, mask=mask)
gray = cv2.cvtColor(mask_white, cv2.COLOR_BGR2GRAY)

# 2
_, binary = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
kernel = np.ones((5, 5), np.uint8)
morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
edges = cv2.Canny(morphed, 40, 120, apertureSize=3)


# 두 이미지를 동시에 표시
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)  # 1행 2열 중 첫 번째
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)  # 1행 2열 중 두 번째
plt.imshow(gray, cmap='gray')
plt.title('Gray Image')
plt.axis('off')

plt.tight_layout()
plt.show()
