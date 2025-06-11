import jh_filename
import os
import shutil

# 버전을 지정해 주세요
version = "v9"  # v10, v11 등으로 바꾸면 적용됨
append_str = f"{version}_"

# 기존 폴더와 새 폴더 경로
source_folder = "output/ODSR_anno"
destination_folder = f"output/ODSR_{version}_anno"

# 기존 폴더 복사
if os.path.isdir(source_folder):
    shutil.copytree(source_folder, destination_folder)
    print(f"'{source_folder}' 폴더를 '{destination_folder}'으로 복사했습니다.")

    # 복사된 폴더 내 모든 파일에 버전 접두어 추가
    jh_filename.prepend_text_to_filenames(destination_folder, append_str)
    print(f"'{destination_folder}' 내 모든 파일에 '{append_str}' 접두어를 추가했습니다.")

else:
    print(f"올바른 폴더 경로를 입력하세요. ({source_folder} 가 존재하지 않음)")

# 기존의 폴더에도 버전 접두어를 추가할지 여부
folder_path = f"output/ODSR_{version}"
if os.path.isdir(folder_path):
    jh_filename.prepend_text_to_filenames(folder_path, append_str)
    print(f"'{folder_path}' 내 모든 파일에 '{append_str}' 접두어를 추가했습니다.")
else:
    print(f"'{folder_path}' 폴더가 존재하지 않아 패스합니다.")