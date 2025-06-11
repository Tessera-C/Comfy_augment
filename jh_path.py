import os
import json

def get_all_image_paths(folder_path):
    """
    Recursively collects all image file paths from the given folder.

    Args:
    folder_path (str): The path to the folder.

    Returns:
    list: A list of paths to image files.
    """
    # Define common image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_paths = []

    # Walk through all directories and subdirectories
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check the file extension
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))

    return image_paths

def get_all_json_paths(folder_path):
    """
    Recursively collects all JSON file paths from the given folder.

    Args:
    folder_path (str): The path to the folder.

    Returns:
    list: A list of paths to JSON files.
    """
    # Define the JSON file extension
    json_extension = '.json'
    json_paths = []

    # Walk through all directories and subdirectories
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check the file extension
            if os.path.splitext(file)[1].lower() == json_extension:
                json_paths.append(os.path.join(root, file))

    return json_paths

def get_unique_labels_from_json(folder_path):
    """
    Extracts unique labels from JSON files in a specified folder.

    Args:
    folder_path (str): Path to the folder containing JSON files.

    Returns:
    set: A set of unique labels found in the JSON files.
    """
    # Get all JSON file paths
    json_paths = get_all_json_paths(folder_path)
    unique_labels = set()

    for json_path in json_paths:
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if "objects" in data:
                    for obj in data["objects"]:
                        if "label" in obj:
                            unique_labels.add(obj["label"])
        except Exception as e:
            print(f"Error reading file {json_path}: {e}")
    
    return unique_labels


def label_from_txt(file_path):
    class_mapping = {
        "road": 0,
        #"sidewalk": 1,
        "parked cars": 2,
        #"rail track": 3,
        "person": 4,
        #"rider": 5,
        "car": 6,
        "truck": 7,
        "bus": 8,
        #"on rails": 9,
        "motorcycle": 10,
        "bicycle": 11,
        "caravan": 12,
        "trailer": 13,
        "building": 14,
        "wall": 15,
        "fence": 16,
        "guard rail": 17,
        "bridge": 18,
        "tunnel": 19,
        "pole": 20,
        #"pole group": 21,
        "traffic sign": 22,
        "traffic light": 23,
        "plants": 24,  # "vegetation": 24,
        # "terrain": 25,
        # "sky": 26,
        # "ground": 27,
        # "dynamic": 28,
        # "static": 29,
    }
    reverse_mapping = {v: k for k, v in class_mapping.items()}

    # 파일이 없으면 빈 문자열 반환
    if not os.path.exists(file_path):
        print(f"파일이 존재하지 않습니다: {file_path}")
        return ""

    # 라인 앞의 인덱스를 추출
    index_set = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                idx = int(parts[0])
                index_set.add(idx)
            except ValueError:
                pass

    # 매핑에 있는 인덱스만 class_names에 추가
    class_names = []
    for i in sorted(index_set):
        if i in reverse_mapping:
            class_names.append(reverse_mapping[i])
        else:
            # 없는 인덱스는 건너뛰기
            pass

    # 리스트 대신 최종 문자열만 반환
    return ", "+", ".join(class_names)


def calculate_polygon_area(polygon):
    """
    Shoelace 공식으로 다각형의 넓이를 계산
    polygon: [[x1, y1], [x2, y2], ..., [xn, yn]] 형태의 좌표 리스트
    return: 다각형의 넓이
    """
    n = len(polygon)
    if n < 3:
        return 0  # 다각형이 아니면 넓이 0

    area = 0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]  # 다음 꼭짓점 (마지막 꼭짓점에서 처음으로 연결)
        area += x1 * y2 - y1 * x2

    return abs(area) / 2  # 절댓값 및 1/2 곱하기


def label_from_json(file_path, label_mapping, area_threshold=10000):
    """
    JSON 파일에서 라벨 이름을 딕셔너리 매핑을 기반으로 추출.
    폴리곤의 넓이가 일정 기준 이하일 경우 해당 라벨은 스킵.
    file_path: JSON 파일 경로
    label_mapping: 라벨 매핑 딕셔너리. {원래 라벨: 매핑 라벨} 형식.
    area_threshold: 폴리곤 넓이의 최소 기준 (기본값: 100)
    return: 매핑된 라벨 목록 (쉼표로 구분된 문자열)
    """
    # 파일이 없으면 빈 문자열 반환
    if not os.path.exists(file_path):
        print(f"파일이 존재하지 않습니다: {file_path}")
        return ""

    # JSON 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 중복 없이 라벨을 저장할 세트
    label_set = set()

    # objects의 각 라벨을 처리
    for obj in data.get('objects', []):
        label = obj.get('label', None)
        polygon = obj.get('polygon', None)
        if label is None or polygon is None:
            continue

        # 매핑 딕셔너리에 없는 라벨은 포함하지 않음
        if label not in label_mapping:
            continue

        # 폴리곤 넓이 계산
        area = calculate_polygon_area(polygon)
        if area < area_threshold:
            continue  # 넓이가 기준 이하인 경우 스킵

        # 매핑된 이름 추가
        label_set.add(label_mapping[label])

    # 결과를 문자열로 반환 (쉼표로 구분)
    return ", " + ", ".join(sorted(label_set))


if __name__ == "__main__":
    folder_path = "gtFine"
    unique_labels = get_unique_labels_from_json(folder_path)
    imgpaths = get_all_image_paths('imagedata/leftImg8bit/train')
    print(len(imgpaths))
    # print(f"Unique labels found: {unique_labels}")
    # print(len(unique_labels))

