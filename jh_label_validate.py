import os
import json
import jh_path
import shutil

def convert_json_to_yolo_segmentation_with_checks(
        json_folder, 
        output_folder, 
        class_mapping,
        expected_img_width=2048,
        expected_img_height=1024
    ):
    """
    Converts JSON files with labeled polygons to YOLO segmentation format, 
    while checking for abnormal labelings:
      - Polygons that exceed image dimensions
      - Duplicated polygons (same label and exact same coordinates)
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    json_files = jh_path.get_all_json_paths(json_folder)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            img_width = data.get('imgWidth', expected_img_width)
            img_height = data.get('imgHeight', expected_img_height)
            objects = data.get('objects', [])
            
            # 1) 체크: 원본 이미지 크기와 불일치하는 경우 (선택적으로 확인)
            if img_width != expected_img_width or img_height != expected_img_height:
                print(f"[WARNING] {json_file}: Image size in JSON ({img_width}x{img_height}) "
                      f"differs from expected size ({expected_img_width}x{expected_img_height}).")

            # 2) 중복 체크를 위해 (label, polygon 좌표 튜플) 저장
            seen_polygons = set()

            yolo_content = []
            for idx, obj in enumerate(objects):
                label = obj.get('label')
                if label not in class_mapping:
                    continue  # Skip unknown labels

                class_index = class_mapping[label]
                polygon = obj.get('polygon', [])

                # 중복 체크를 위해 polygon 좌표를 tuple로 만들기
                polygon_tuple = tuple(tuple(point) for point in polygon)

                # 중복 라벨링 감지
                if (label, polygon_tuple) in seen_polygons:
                    print(f"[DUPLICATE] {json_file} -> Object Index {idx}, Label: '{label}' "
                          f"has exact duplicate polygon.")
                else:
                    seen_polygons.add((label, polygon_tuple))

                # YOLO Segmentation 포맷에 맞춰 좌표 정규화
                normalized_polygon = []
                for point_idx, point in enumerate(polygon):
                    x_abs, y_abs = point
                    # 범위 체크
                    if x_abs < 0 or x_abs > img_width or y_abs < 0 or y_abs > img_height:
                        print(f"[OUT OF RANGE] {json_file} -> Object Index {idx}, Label: '{label}', "
                              f"Point {point_idx} has coordinates ({x_abs}, {y_abs}) "
                              f"outside image size ({img_width}, {img_height}).")

                    x = x_abs / img_width
                    y = y_abs / img_height

                    # Clamp values to the range [0, 1]
                    x = max(0, min(1, x))
                    y = max(0, min(1, y))

                    normalized_polygon.append(f"{x:.6f} {y:.6f}")

                # 하나의 줄로 class_index + 폴리곤 좌표
                yolo_row = f"{class_index} " + " ".join(normalized_polygon)
                yolo_content.append(yolo_row)

            # 최종적으로 YOLO 형식의 txt 파일로 저장
            txt_file_name = os.path.splitext(os.path.basename(json_file))[0] + ".txt"
            txt_file_path = os.path.join(output_folder, txt_file_name)
            with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write("\n".join(yolo_content))

        except Exception as e:
            print(f"Error processing {json_file}: {e}")


def move_files_to_folder(file_paths, output_folder):
    """
    Moves files from the provided paths to the output folder.

    Args:
    file_paths (list): List of file paths to move.
    output_folder (str): Path to the folder where files will be moved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                # Get the file name
                file_name = os.path.basename(file_path)
                # Define the destination path
                destination = os.path.join(output_folder, file_name)
                # Move the file
                shutil.move(file_path, destination)
                print(f"Moved: {file_path} -> {destination}")
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error moving {file_path}: {e}")

def replace_part_in_filenames(folder_path, old_part, new_part):
    """
    Replaces a specific part of file names in the given folder.

    Args:
    folder_path (str): Path to the folder containing the files.
    old_part (str): The part of the file name to replace.
    new_part (str): The new part to replace the old part with.
    """
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return

    # List all files in the folder
    files = os.listdir(folder_path)

    for file_name in files:
        # Construct full file path
        old_file_path = os.path.join(folder_path, file_name)

        # Skip directories
        if not os.path.isfile(old_file_path):
            continue

        # Replace the old part with the new part in the file name
        new_file_name = file_name.replace(old_part, new_part)
        new_file_path = os.path.join(folder_path, new_file_name)

        # Rename the file
        try:
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} -> {new_file_path}")
        except Exception as e:
            print(f"Error renaming {old_file_path}: {e}")


if __name__ == "__main__":
    # Example usage
    json_folder = 'input/Image_anno/train'
    output_folder = 'yolo_train1'

    # Mapping of class names to YOLO indices
    class_mapping = {
        "road": 0,
        "sidewalk": 1,
        "parking": 2,
        "rail track": 3,
        "person": 4,
        "rider": 5,
        "car": 6,
        "truck": 7,
        "bus": 8,
        "on rails": 9,
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
        "pole group": 21,
        "traffic sign": 22,
        "traffic light": 23,
        "vegetation": 24,
        "terrain": 25,
        "sky": 26,
        "ground": 27,
        "dynamic": 28,
        "static": 29,
    }
    
    # 2048x1024이 기대되는 이미지 크기 (Cityscapes 기준) 
    convert_json_to_yolo_segmentation_with_checks(
        json_folder=json_folder, 
        output_folder=output_folder, 
        class_mapping=class_mapping,
        expected_img_width=2048,
        expected_img_height=1024
    )

    # (예) 이미지 파일 옮기기
    # imgpaths = jh_path.get_all_image_paths('imagedata/leftImg8bit/val')
    # move_files_to_folder(imgpaths, 'val')

    # (예) 파일명에서 gtFine_polygons -> leftImg8bit 변경하기
    # folder_path = "yolo_train"
    # old_part = "gtFine_polygons"
    # new_part = "leftImg8bit"
    # replace_part_in_filenames(folder_path, old_part, new_part)