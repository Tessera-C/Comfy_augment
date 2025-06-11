import os
import numpy as np
import cv2
import tifffile as tiff
from skimage.measure import find_contours

# Cityscapes 데이터 경로
panoptic_path = "aachen_000000_000019_gtFinePanopticParts.tif"
# image_path = "cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"
output_label_path = "yolo_labels/aachen_000000_000019.txt"

# TIFF 파일 로드
panoptic_img = tiff.imread(panoptic_path)
class_mask = panoptic_img // 1000  # Semantic Segmentation용 class mask
instance_mask = panoptic_img % 1000  # Instance Segmentation용 mask

# 원본 이미지 크기 로드
# image = cv2.imread(image_path)
height = 1024
width = 2048

# YOLO Segmentation 라벨링 저장
os.makedirs("yolo_labels", exist_ok=True)
with open(output_label_path, "w") as f:
    unique_instances = np.unique(instance_mask)
    
    for instance_id in unique_instances:
        if instance_id == 0:  # 배경 제외
            continue

        # 객체가 있는 위치 찾기
        obj_mask = (instance_mask == instance_id).astype(np.uint8)

        # Contour 검출 (YOLO 포맷을 위해 Polygon 필요)
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 폴리곤 좌표를 YOLO 포맷으로 변환
            polygon = []
            for point in contour:
                x, y = point[0]
                polygon.append(x / width)
                polygon.append(y / height)

            # 클래스 ID 가져오기
            class_id = int(class_mask[obj_mask > 0][0])  # 객체 내부의 클래스 ID
            
            # YOLO 형식으로 저장
            line = f"{class_id} " + " ".join(map(str, polygon)) + "\n"
            f.write(line)

print(f"YOLO Segmentation 라벨 생성 완료: {output_label_path}")
