import os
import glob

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
            images.append(image_file)

# 35,564
print(len(images))