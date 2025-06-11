import json
import cv2
import numpy as np
import random
import matplotlib
# matplotlib.use('Agg')  # GUI를 사용하지 않는 백엔드 설정
import matplotlib.pyplot as plt

def visualize_labels_individually(json_path, img_path=None, alpha=0.5):
    """
    JSON 파일로부터 라벨들을 하나씩 오버레이하여 보여주는 함수.
    ESC 키를 누르면 중도 종료, 다른 키를 누르면 다음 라벨로 넘어갑니다.

    Args:
        json_path (str): JSON 파일 경로
        img_path (str, optional): 배경 이미지 파일 경로 (없으면 검은 배경)
        alpha (float, optional): 오버레이 투명도 (기본 0.5)
    """
    # JSON 데이터 불러오기
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    img_height = json_data['imgHeight']
    img_width = json_data['imgWidth']

    # 배경 이미지 불러오기 (또는 빈 캔버스 만들기)
    if img_path is not None:
        background = cv2.imread(img_path)
        if background is None:
            raise FileNotFoundError(f"Image file not found: {img_path}")
        background = cv2.resize(background, (img_width, img_height))
    else:
        background = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # 객체 정보 가져오기
    objects = json_data.get('objects', [])
    
    # 라벨별 색상 사전 (없으면 랜덤)
    label_colors = {
        "road": (128, 64, 128),
        "sidewalk": (244, 35, 232),
        "car": (0, 0, 142),
        "sky": (70, 130, 180),
        "terrain": (152, 251, 152),
        "building": (70, 70, 70),
        "vegetation": (107, 142, 35),
        "pole": (153, 153, 153),
        "traffic sign": (220, 220, 0),
        "static": (0, 128, 128),
        "traffic light": (194, 43, 137),
        "cargroup": (60, 221, 144),
        "person": (117, 84, 207),
        "dynamic": (85, 88, 158),
        "ego vehicle": (167, 69, 215),
        "rectification border": (115, 134, 27),
        "out of roi": (28, 197, 156),
    }
    
    # JSON에 등장하는 유니크한 라벨 목록 추출
    unique_labels = set(obj['label'] for obj in objects)
    unique_labels = sorted(list(unique_labels))  # 보기 좋게 정렬
    
    print(f"총 라벨 개수: {len(unique_labels)}개")
    print("ESC 키를 누르면 중단합니다. 다른 키를 누르면 다음 라벨로 넘어갑니다.")
    
    for label in unique_labels:
        # 배경 복사본 생성
        bg_copy = background.copy()
        # 오버레이용 캔버스
        overlay = np.zeros_like(bg_copy)
        
        # 현재 라벨을 갖는 객체만 오버레이
        for obj in objects:
            if obj['label'] == label:
                polygon = np.array(obj['polygon'], dtype=np.int32)
                
                # 만약 라벨 색상이 label_colors에 없으면 랜덤 생성
                if label not in label_colors:
                    label_colors[label] = tuple(random.randint(0, 255) for _ in range(3))
                
                # 폴리곤 채우기
                cv2.fillPoly(overlay, [polygon], label_colors[label])
                
                # 폴리곤의 바운딩 박스 계산 (라벨을 붙이기 위해)
                x, y, w, h = cv2.boundingRect(polygon)
                # 바운딩 박스 위치에 라벨 텍스트 표시
                text_pos = (x, max(y - 5, 0))  # 약간 위쪽에 표시
                cv2.putText(
                    overlay, 
                    label, 
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (255, 255, 255), 2, 
                    cv2.LINE_AA
                )
        
        # 투명도 적용
        blended = cv2.addWeighted(overlay, alpha, bg_copy, 1 - alpha, 0)
        
        # 현재 라벨 크게 표시 (화면 왼쪽 상단)
        label_text = f"Label: {label}"
        cv2.putText(
            blended, 
            label_text, 
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.5, 
            (0, 255, 0), 3, 
            cv2.LINE_AA
        )
        
        # 이미지 보여주기
        cv2.imshow("Label Visualization", blended)
        
        # 키 입력 대기 (0이면 무한 대기)
        key = cv2.waitKey(0)
        # ESC 키(27)를 누르면 종료
        if key == 27:  # ESC
            break
    
    cv2.destroyAllWindows()


def visualize_and_save_segmentation(
    json_path, 
    img_path=None, 
    output_path="segmentation_result.png", 
    alpha=0.5, 
    exclude_labels=None
):
    """
    JSON 데이터와 선택적으로 이미지를 입력으로 받아 객체를 시각화하고 결과 이미지를 파일로 저장하는 함수.

    Args:
        json_path (str): JSON 파일 경로
        img_path (str, optional): 배경 이미지 파일 경로. None이면 빈 캔버스를 사용
        output_path (str): 저장할 출력 이미지 파일 경로
        alpha (float): 시각화의 투명도 (0.0 - 1.0)
        exclude_labels (list, optional): 시각화에서 제외할 라벨 목록
    """
    # 제외할 라벨 기본값 설정
    if exclude_labels is None:
        exclude_labels = []

    # JSON 파일 읽기
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    img_height = json_data['imgHeight']
    img_width = json_data['imgWidth']

    # 이미지 로드 또는 빈 캔버스 생성
    if img_path:
        background = cv2.imread(img_path)
        if background is None:
            raise FileNotFoundError(f"Image file not found at {img_path}")
        background = cv2.resize(background, (img_width, img_height))  # 이미지 크기 맞춤
    else:
        background = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # 객체별 색상 매핑 (업데이트된 정의)
    label_colors = {
        "road": (128, 64, 128),
        "sidewalk": (244, 35, 232),
        "car": (0, 0, 142),
        "sky": (70, 130, 180),
        "terrain": (152, 251, 152),
        "building": (70, 70, 70),
        "vegetation": (107, 142, 35),
        "pole": (153, 153, 153),
        "traffic sign": (220, 220, 0),
        "static": (0, 128, 128),
        "traffic light": (194, 43, 137),
        "cargroup": (60, 221, 144),
        "person": (117, 84, 207),
        "dynamic": (85, 88, 158),
        "ego vehicle": (167, 69, 215),
        "rectification border": (115, 134, 27),
        "out of roi": (28, 197, 156),
    }

    # 투명도 적용을 위해 빈 캔버스 생성
    overlay = np.zeros_like(background)

    # 객체들을 캔버스에 그리기
    for obj in json_data['objects']:
        label = obj['label']
        
        # 제외할 라벨인지 확인
        if label in exclude_labels:
            continue  # 제외 라벨이면 건너뛰기

        polygon = np.array(obj['polygon'], dtype=np.int32)

        # 라벨에 색상이 없으면 랜덤 색상 생성
        if label not in label_colors:
            label_colors[label] = tuple(random.randint(0, 255) for _ in range(3))

        # 라벨 색상 가져오기
        color = label_colors[label]
        cv2.fillPoly(overlay, [polygon], color)

    # 배경과 오버레이를 합성 (투명도 적용)
    blended = cv2.addWeighted(overlay, alpha, background, 1 - alpha, 0)

    # 결과 이미지를 파일로 저장
    cv2.imwrite(output_path, blended)

    print(f"Segmentation result saved to {output_path}")
    return label_colors  # 라벨별 색상 반환

if __name__ == "__main__":
    json_path = "frankfurt_000000_000294_gtFine_polygons.json"
    img_path = "frankfurt_000000_000294_leftImg8bit.png"
    
    visualize_labels_individually(json_path, img_path, alpha=0.5)

# # 사용 예시
# json_path = "frankfurt_000000_002196_gtFine_polygons.json"
# img_path = "frankfurt_000000_002196_leftImg8bit.png"
# output_path = "visual1.png"
# exclude_labels = []  # 제외할 라벨 설정

# # 함수 호출
# label_colors = visualize_and_save_segmentation(
#     json_path, img_path, output_path, alpha=0.5, exclude_labels=exclude_labels
# )

# # 라벨별 색상 출력
# print("Assigned Colors for Labels:")
# for label, color in label_colors.items():
#     print(f"{label}: {color}")
