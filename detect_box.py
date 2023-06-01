import cv2
import numpy as np
from PIL import Image

def find_green_boxes(image):
    # 이미지를 HSV 색 공간으로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 초록색 범위 지정 (Hue: 60-90, Saturation: 100-255, Value: 50-255)
    # lower_green = np.array([60, 150, 50])
    # upper_green = np.array([70, 255, 255])
    lower_green = np.array([35, 160, 160])
    upper_green = np.array([85, 255, 255])
    
    # 초록색 영역을 찾기 위한 임계값 처리
    mask = cv2.inRange(hsv, lower_green, upper_green)

    return mask

    # for contour in contours:
    #     # 윤곽선을 감싸는 최소한의 사각형 경계 상자 추출
    #     x, y, w, h = cv2.boundingRect(contour)

    #     # 경계 상자를 초록색 박스 리스트에 추가
    #     green_boxes.append((x, y, w, h))

    # return green_boxes


# 이미지 로드
image_path = 'F002-0004.jpg'
image = cv2.imread(image_path)
image_np = np.array(image)
print(image_np.shape)
# 492x658

# 초록색 박스 검출
green_boxes = find_green_boxes(image)

result = np.zeros([492,658, 3])
# print(result.shape)

x_boundary = []
y_boundary = []

for y, y_img in enumerate(green_boxes):
    for x, x_img in enumerate(y_img):
        if x_img != 0:
            x_boundary.append(x)
            y_boundary.append(y)
            result[y, x, :] = image_np[y, x, :]

x_min = min(x_boundary)
x_max = max(x_boundary)

y_min = min(y_boundary)
y_max = max(y_boundary)

crop_img = image_np[y_min:y_max, x_min:x_max, :]

output_path = 'green_image.jpg'
cv2.imwrite(output_path, result)
output_path = 'cropped_image.jpg'
cv2.imwrite(output_path, crop_img)