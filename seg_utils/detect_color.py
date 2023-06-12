import cv2
import numpy as np

# 이미지를 읽어옵니다.
image = cv2.imread('F002-0004.jpg')

# 이미지를 BGR에서 HSV로 변환합니다.
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 파란색 범위를 정의합니다. (색상, 채도, 명도)
# lower_blue = np.array([90, 50, 50])
# upper_blue = np.array([130, 255, 255])
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

# 파란색 영역을 추출합니다.
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# 파란색 영역만 남기고 나머지는 검정색으로 채웁니다.
blue_only = cv2.bitwise_and(image, image, mask=blue_mask)
output_path = 'blue_image.jpg'
cv2.imwrite(output_path, blue_only)

# 빨간색 범위를 정의합니다. (색상, 채도, 명도)
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([20, 255, 255])
lower_red2 = np.array([160, 50, 50])
upper_red2 = np.array([190, 255, 255])

# 빨간색 영역을 추출합니다.
red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(red_mask1, red_mask2)

# 빨간색 영역만 남기고 나머지는 검정색으로 채웁니다.
red_only = cv2.bitwise_and(image, image, mask=red_mask)
output_path = 'red_image.jpg'
cv2.imwrite(output_path, red_only)

# 초록색 범위를 정의합니다. (색상, 채도, 명도)
lower_green = np.array([35, 160, 160])
upper_green = np.array([85, 255, 255])

# 초록색 영역을 추출합니다.
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
# 초록색 영역만 남기고 나머지는 검정색으로 채웁니다.
green_only = cv2.bitwise_and(image, image, mask=green_mask)
output_path = 'green_image.jpg'
cv2.imwrite(output_path, green_only)