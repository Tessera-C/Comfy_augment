import os
import re
import shutil

def rename_images_in_folder(folder_path):
    """
    기존에 사용하던 함수.
    폴더 내의 이미지 파일 중 '_00001_' 등 5자리 숫자 패턴이 있는 경우 이를 제거.
    """
    # 이미지 확장자 목록
    image_extensions = ['.png', '.jpg', '.jpeg']

    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            # '_00001_' 또는 '_00002_' 등 5자리 숫자 패턴 제거
            new_filename = re.sub(r'_\d{5}_', '_', filename)

            if filename != new_filename:
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} → {new_filename}")
            else:
                print(f"No change needed: {filename}")


def copy_files_of_extensions_to_single_folder(source_folder, target_folder, extensions):
    """
    source_folder 아래 (하위 폴더까지) 있는 모든 파일을 탐색하여
    지정한 확장자(extensions)에 해당하는 파일을 target_folder로 '복사'합니다.

    :param source_folder: 재귀적으로 탐색할 폴더
    :param target_folder: 파일을 모을 폴더 (하위 폴더 없이 전부 이 폴더에 복사)
    :param extensions: 찾고자 하는 확장자 목록 (예: ['.png', '.jpg', '.json'])
    """
    # target_folder가 없으면 생성
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # os.walk를 통해 source_folder를 재귀적으로 순회
    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()  # 확장자 추출
            if ext in extensions:
                source_path = os.path.join(root, filename)
                target_path = os.path.join(target_folder, filename)
                
                # shutil.copy2를 사용하면 메타데이터(수정일 등)까지 복사합니다.
                shutil.copy2(source_path, target_path)
                print(f"Copied: {source_path} → {target_path}")

def prepend_text_to_filenames(folder_path, text_to_prepend):
    """
    folder_path 안에 존재하는 모든 파일에 대해
    파일 이름 앞에 text_to_prepend를 추가합니다.
    예) example.png -> _v2example.png

    :param folder_path: 처리할 폴더의 경로
    :param text_to_prepend: 파일명 앞에 추가할 텍스트 (예: '_v2')
    """
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)

        # 일반 파일(디렉터리 제외)만 처리
        if os.path.isfile(old_path):
            # 확장자 분리
            name, ext = os.path.splitext(filename)

            # 새 파일명 생성 (텍스트 + 기존 파일 이름 + 확장자)
            new_name = text_to_prepend + name + ext
            new_path = os.path.join(folder_path, new_name)

            # 파일 이름 변경
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_name}")

def revert_appended_text_in_filenames(folder_path, appended_text):
    """
    folder_path 안에 존재하는 모든 파일에 대해
    파일 이름에 추가된 appended_text를 제거합니다.
    예) example_v2.png -> example.png

    :param folder_path: 처리할 폴더의 경로
    :param appended_text: 제거할 텍스트 (예: '_v2')
    """
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)

        if os.path.isfile(old_path):
            name, ext = os.path.splitext(filename)

            # 추가된 텍스트가 파일명 끝에 정확히 붙어 있는 경우만 제거
            if name.endswith(appended_text):
                # 예: '_v2'의 길이만큼 잘라내기
                new_name = name[: -len(appended_text)] + ext
                new_path = os.path.join(folder_path, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} → {new_name}")


def revert_prepended_text_in_filenames(folder_path, prepended_text):
    """
    folder_path 안에 존재하는 모든 파일에 대해
    파일 이름 앞에 추가된 prepended_text를 제거합니다.
    예) v2_example.png -> example.png

    :param folder_path: 처리할 폴더의 경로
    :param prepended_text: 제거할 텍스트 (예: 'v2_')
    """
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)

        if os.path.isfile(old_path):
            name, ext = os.path.splitext(filename)

            # 앞에 붙은 텍스트가 있는 경우만 제거
            if name.startswith(prepended_text):
                new_name = name[len(prepended_text):] + ext
                new_path = os.path.join(folder_path, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} → {new_name}")



if __name__ == "__main__":
    # 예시 사용 방법

    # 1) 특정 폴더에서 '_00001_' 등 제거 후 이름 정리
    # folder_to_rename = "output/cityscapes2"
    # if os.path.isdir(folder_to_rename):
    #     rename_images_in_folder(folder_to_rename)
    #     print("모든 이미지 파일의 이름이 정리되었습니다.")
    # else:
    #     print("올바른 폴더 경로를 입력하세요.")




    # 2) 확장자별로 파일을 모으기
    # source_folder = "input/Image_anno"
    # target_folder = "output/annotations"
    # extensions_to_find = ['.png']


    # if os.path.isdir(source_folder):
    #     copy_files_of_extensions_to_single_folder(source_folder, target_folder, extensions_to_find)
    #     print(f"{source_folder} 내 모든 {extensions_to_find} 파일을 {target_folder}로 이동했습니다.")
    # else:
    #     print("올바른 폴더 경로를 입력하세요.")




    folder_path = "output/ODSR_v8_anno"
    append_str = "v8_"

    if os.path.isdir(folder_path):
        prepend_text_to_filenames(folder_path, append_str)
        print(f"'{folder_path}' 폴더 내 모든 파일에 '{append_str}'가 추가되었습니다.")
    else:
        print("올바른 폴더 경로를 입력하세요.")


    # revert_prepended_text_in_filenames(folder_path, append_str)
    # print(f"'{folder_path}' 내 파일에서 '{append_str}'가 제거되었습니다.")