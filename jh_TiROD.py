import json
from collections import defaultdict
from typing import Dict

def load_annotation_data(json_path: str):
    """val.json 파일을 불러오고 필요한 맵핑 정보를 준비합니다."""
    with open(json_path, "r") as f:
        data = json.load(f)

    image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    filename_to_image_id = {v: k for k, v in image_id_to_filename.items()}
    category_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}

    image_annotations = defaultdict(list)
    for ann in data["annotations"]:
        image_annotations[ann["image_id"]].append(ann)

    return filename_to_image_id, category_id_to_name, image_annotations

def get_object_counts_for_image(image_filename: str,
                                 filename_to_image_id: Dict[str, int],
                                 category_id_to_name: Dict[int, str],
                                 image_annotations: Dict[int, list]) -> Dict[str, int]:
    """이미지 파일 이름을 받아 객체 수를 카테고리별로 반환합니다."""
    if image_filename not in filename_to_image_id:
        raise ValueError(f"이미지 파일 '{image_filename}'을 찾을 수 없습니다.")

    image_id = filename_to_image_id[image_filename]
    annotations = image_annotations.get(image_id, [])

    object_count = defaultdict(int)
    for ann in annotations:
        category_name = category_id_to_name[ann["category_id"]]
        object_count[category_name] += 1
    print(dict(object_count))
    return dict(object_count)
    

def print_object_summary(image_filename: str,
                         filename_to_image_id: Dict[str, int],
                         category_id_to_name: Dict[int, str],
                         image_annotations: Dict[int, list]):
    """객체 수를 콘솔에 출력합니다."""
    try:
        counts = get_object_counts_for_image(image_filename, filename_to_image_id, category_id_to_name, image_annotations)
        if not counts:
            print(f"이미지 '{image_filename}'에는 객체 정보가 없습니다.")
            return
        print(f"이미지 '{image_filename}' 객체 요약:")
        for category, count in counts.items():
            print(f"- {category}: {count}개")
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    # 1. 먼저 데이터를 로드
    filename_to_image_id, category_id_to_name, image_annotations = load_annotation_data("input/TiROD/Domain1/High/annotations/train.json")

    # 2. 특정 이미지에 대해 요약 출력
    print_object_summary("d1_h_frame78.png", filename_to_image_id, category_id_to_name, image_annotations)