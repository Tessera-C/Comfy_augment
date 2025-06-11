import os
import json
import jh_path
import shutil

def convert_json_to_yolo_segmentation(json_folder, output_folder, class_mapping):
    """
    Converts JSON files with labeled polygons to YOLO segmentation format.

    Args:
    json_folder (str): Path to the folder containing JSON files.
    output_folder (str): Path to the folder where YOLO txt files will be saved.
    class_mapping (dict): Dictionary mapping class names to YOLO class indices.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    json_files = jh_path.get_all_json_paths(json_folder)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            img_width = data['imgWidth']
            img_height = data['imgHeight']
            objects = data.get('objects', [])

            # Prepare YOLO text file content
            yolo_content = []
            for obj in objects:
                label = obj.get('label')
                if label not in class_mapping:
                    continue  # Skip unknown labels

                class_index = class_mapping[label]
                polygon = obj.get('polygon', [])

                # Normalize polygon coordinates
                normalized_polygon = []
                for point in polygon:
                    x = point[0] / img_width
                    y = point[1] / img_height
                    
                    # Clamp values to the range [0, 1]
                    x = max(0, min(1, x))
                    y = max(0, min(1, y))
                    
                    normalized_polygon.append(f"{x:.6f} {y:.6f}")

                # Combine class index and normalized polygon
                yolo_row = f"{class_index} " + " ".join(normalized_polygon)
                yolo_content.append(yolo_row)

            # Save to YOLO txt file
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
    json_folder = 'gtFine/train'
    output_folder = 'yolo_train'

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
    imgpaths = jh_path.get_all_image_paths('imagedata/leftImg8bit/val')
    # convert_json_to_yolo_segmentation(json_folder, output_folder, class_mapping)
    # move_files_to_folder(imgpaths, 'val')
    folder_path = "yolo_train"
    old_part = "gtFine_polygons"
    new_part = "leftImg8bit"
    replace_part_in_filenames(folder_path, old_part, new_part)