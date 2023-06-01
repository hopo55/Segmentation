import os
import cv2
import glob
import numpy as np

def find_color(image, lower1, upper1, lower2=None, upper2=None):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 초록색 영역을 찾기 위한 임계값 처리
    if lower2 is None:
        mask = cv2.inRange(hsv_image, lower1, upper1)
        color_only = cv2.bitwise_and(image, image, mask=mask)
    else:
        mask1 = cv2.inRange(hsv_image, lower1, upper1)
        mask2 = cv2.inRange(hsv_image, lower2, upper2)
        color_only = cv2.bitwise_or(mask1, mask2)
    
    return color_only

def cut_image(image, boundary):
    image_np = np.array(image)
    cut_img = np.zeros(image_np.shape)
    x_boundary = []
    y_boundary = []

    for y, y_img in enumerate(boundary):
        for x, x_img in enumerate(y_img):
            if not np.array_equal(x_img, [0, 0, 0]):
                x_boundary.append(x)
                y_boundary.append(y)
                cut_img[y, x, :] = image_np[y, x, :]

    x_min = min(x_boundary)
    x_max = max(x_boundary)

    y_min = min(y_boundary)
    y_max = max(y_boundary)

    cut_img = image_np[y_min:y_max, x_min:x_max, :]

    return cut_img

def get_target(image_path):
    image = cv2.imread(image_path)
    _, folder_name, file_name = image_path.split("/")

    # Green
    lower_green = np.array([35, 160, 160])
    upper_green = np.array([85, 255, 255])

    green_mask = find_color(image, lower_green, upper_green)
    if np.count_nonzero(green_mask) < 10:
        return print(folder_name + '/' + file_name + ' pass')

    cut_img = cut_image(image, green_mask)
    output_path = 'green_image.jpg'
    cv2.imwrite(output_path, cut_img)

    # Blue
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    blue_mask = find_color(cut_img, lower_blue, upper_blue)

    if not os.path.exists('label/' + folder_name):
        os.makedirs('label/' + folder_name)

    output_path = 'label/' + folder_name + '/' + file_name
    cv2.imwrite(output_path, blue_mask)

# 상위 폴더 경로
folder_path = "data"
images = []

# 모든 하위 폴더 순회
for root, dirs, files in os.walk(folder_path):
    for dir in dirs:
        # 하위 폴더 내의 이미지 파일 목록 가져오기
        subfolder_path = os.path.join(root, dir)
        image_files = glob.glob(subfolder_path + "/*.jpg")  # .jpg 확장자를 가진 이미지 파일들을 가져옴

        # 이미지 파일들을 읽어오기
        for image_file in image_files:
            get_target(image_file)

