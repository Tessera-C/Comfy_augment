import json
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def visualize_coco_bbox(img_path, ann_path, box_color=(255, 0, 0), text_color=(255, 0, 0),
                        box_thickness=2, text_scale=0.6, text_thickness=1, show=True):
    """
    COCO annotation을 기반으로 이미지에 bounding box 시각화

    Parameters:
        img_path (str or Path): 이미지 경로
        ann_path (str or Path): COCO json 파일 경로
        box_color (tuple): bounding box 색 (BGR)
        text_color (tuple): 텍스트 색 (BGR)
        box_thickness (int): bounding box 선 두께
        text_scale (float): 텍스트 크기
        text_thickness (int): 텍스트 두께
        show (bool): matplotlib으로 시각화 여부

    Returns:
        image (numpy.ndarray): bounding box가 그려진 이미지
    """
    img_path = Path(img_path)
    ann_path = Path(ann_path)

    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width = image.shape[:2]

    with open(ann_path, 'r') as f:
        coco = json.load(f)

    category_map = {cat['id']: cat['name'] for cat in coco['categories']}

    image_id = next((img['id'] for img in coco['images'] if img['file_name'] == img_path.name), None)
    if image_id is None:
        raise ValueError(f"Image {img_path.name} not found in annotation file.")

    annotations = [ann for ann in coco['annotations'] if ann['image_id'] == image_id]

    for ann in annotations:
        x, y, w, h = map(int, ann['bbox'])
        category_id = ann['category_id']
        category_name = category_map.get(category_id, "unknown")

        # bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), box_color, box_thickness)

        # 텍스트 위치
        text_y = y - 5 if y > 20 else y + h + 15
        cv2.putText(image, category_name, (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, text_thickness)

    if show:
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Annotations for {img_path.name}")
        plt.show()

    return image



def create_mask_from_coco(img_path, ann_path, output_mask_path):
    """
    COCO 형식 annotation으로부터 bounding box 영역만 흰색인 마스크 이미지 생성

    Parameters:
        img_path (str or Path): 원본 이미지 경로
        ann_path (str or Path): COCO annotation JSON 파일 경로
        output_mask_path (str or Path): 저장할 마스크 이미지 경로 (PNG 권장)
    """
    img_path = Path(img_path)
    ann_path = Path(ann_path)
    output_mask_path = Path(output_mask_path)

    # 원본 이미지 크기 얻기
    img = cv2.imread(str(img_path))
    height, width = img.shape[:2]

    # 빈 마스크 생성
    mask = np.zeros((height, width), dtype=np.uint8)

    # COCO annotation 로딩
    with open(ann_path, 'r') as f:
        coco = json.load(f)

    # 현재 이미지에 해당하는 image_id 찾기
    image_id = next((img['id'] for img in coco['images'] if img['file_name'] == img_path.name), None)
    if image_id is None:
        raise ValueError(f"Image {img_path.name} not found in annotation file.")

    # 해당 image_id의 annotation만 선택
    annotations = [ann for ann in coco['annotations'] if ann['image_id'] == image_id]

    # bounding box마다 사각형을 흰색으로 채움
    for ann in annotations:
        x, y, w, h = map(int, ann['bbox'])
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # 저장
    cv2.imwrite(str(output_mask_path), mask)
    print(f"Mask saved to {output_mask_path}")



def create_rgba_inpaint_image(img_path, ann_path, output_path, target_classes=None):
    """
    COCO bounding box를 기반으로 RGBA 이미지 생성.
    특정 클래스만 투명(알파=0), 나머지는 불투명(알파=255)

    Parameters:
        img_path (str or Path): 원본 이미지 경로
        ann_path (str or Path): COCO annotation JSON 파일 경로
        output_path (str or Path): 저장할 RGBA 이미지 경로 (PNG)
        target_classes (list of str or None): 마스크 처리할 클래스 이름 리스트. None이면 전체 처리
    """
    img_path = Path(img_path)
    ann_path = Path(ann_path)
    output_path = Path(output_path)

    # 원본 이미지 불러오기 (RGB)
    image_bgr = cv2.imread(str(img_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    # 알파 채널: 전체 불투명(255)
    alpha_channel = np.full((height, width), 255, dtype=np.uint8)

    # COCO 로딩
    with open(ann_path, 'r') as f:
        coco = json.load(f)

    # 클래스 ID → 이름 매핑
    category_map = {cat['id']: cat['name'] for cat in coco['categories']}

    # 현재 이미지 ID 찾기
    image_id = next((img['id'] for img in coco['images'] if img['file_name'] == img_path.name), None)
    if image_id is None:
        raise ValueError(f"{img_path.name} not found in annotation.")

    # 해당 이미지의 annotation
    annotations = [ann for ann in coco['annotations'] if ann['image_id'] == image_id]

    for ann in annotations:
        category_id = ann['category_id']
        category_name = category_map.get(category_id, "unknown")

        # 클래스 필터링: 전체 또는 선택된 클래스만 처리
        if target_classes is None or category_name in target_classes:
            x, y, w, h = map(int, ann['bbox'])
            alpha_channel[y:y+h, x:x+w] = 0

    # RGBA 이미지 생성
    rgba_image = np.dstack((image_rgb, alpha_channel))

    # 저장
    cv2.imwrite(str(output_path), cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA))
    print(f"RGBA inpaint image saved to {output_path}")

if __name__ == "__main__":
    # 예시 실행 1
    # img_path = "d1_h_frame752.png"
    # ann_path = "train.json"
    # visualize_coco_bbox(img_path, ann_path)

    # example run 2
    # create_mask_from_coco(
    # img_path="d1_h_frame752.png",
    # ann_path="train.json",
    # output_mask_path="d1_h_frame752_mask.png")

    # exmaple run 3
    if __name__ == "__main__":
        create_rgba_inpaint_image(
            img_path="d1_h_frame752.png",
            ann_path="train.json",
            output_path="rgba_chair_only.png",
            target_classes=["chair"]
        )