import cv2
import os

def crop_image_by_pixels(image_path, output_path, crop_left, crop_top, crop_right, crop_bottom):
    """
    이미지를 불러와서 가장자리에서 지정한 픽셀만큼 잘라내는 함수.
    
    :param image_path: 수정할 이미지 경로 (png 파일)
    :param output_path: 결과 이미지를 저장할 경로
    :param crop_left: 왼쪽에서 잘라낼 픽셀 수
    :param crop_top: 위쪽에서 잘라낼 픽셀 수
    :param crop_right: 오른쪽에서 잘라낼 픽셀 수
    :param crop_bottom: 아래쪽에서 잘라낼 픽셀 수
    """
    # 이미지 읽기
    img = cv2.imread(image_path)

    if img is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    # 원본 이미지 크기 확인
    height, width, _ = img.shape

    # 잘라낼 크기 제한 (이미지 크기보다 많이 잘라내지 않도록 조정)
    crop_left = min(crop_left, width // 2)
    crop_right = min(crop_right, width // 2)
    crop_top = min(crop_top, height // 2)
    crop_bottom = min(crop_bottom, height // 2)

    # 자르기 위한 좌표 정의
    left = crop_left
    upper = crop_top
    right = width - crop_right
    lower = height - crop_bottom

    # 이미지 자르기
    cropped_img = img[upper:lower, left:right]

    # 결과 디렉토리가 존재하지 않으면 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 결과 이미지 저장
    cv2.imwrite(output_path, cropped_img)
    print(f"이미지가 성공적으로 {output_path}에 저장되었습니다.")

# 예시 사용: 왼쪽에서 50픽셀, 위쪽에서 30픽셀, 오른쪽에서 50픽셀, 아래쪽에서 30픽셀 잘라내기
if __name__ == "__main__":
    import jh_path
    
    input_paths = jh_path.get_all_image_paths('input/cityscapes_left8bit')
    output_paths = [path.replace("input/", "image_processed/") for path in input_paths]

    for in_p, out_p in zip(input_paths, output_paths):
        crop_image_by_pixels(in_p, out_p, crop_left=50, crop_top=30, crop_right=50, crop_bottom=30)
