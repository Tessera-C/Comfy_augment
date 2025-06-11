import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

# TIFF 파일 불러오기
panoptic_path = "aachen_000000_000019_gtFinePanopticParts.tif"
panoptic_img = tiff.imread(panoptic_path)  # 2D 배열

# Semantic Class Map (픽셀 값을 1000으로 나눈 몫)
semantic_map = panoptic_img // 1000

# Instance ID Map (픽셀 값을 1000으로 나눈 나머지)
instance_map = panoptic_img % 1000

# 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Semantic Segmentation")
plt.imshow(semantic_map, cmap="jet")

plt.subplot(1, 2, 2)
plt.title("Instance Segmentation")
plt.imshow(instance_map, cmap="jet")

plt.show()

from PIL import Image

# PNG 저장 (Semantic Map)
semantic_img = Image.fromarray((semantic_map * 10).astype(np.uint8))  # 값이 작아서 10배 확대
semantic_img.save("semantic_map.png")

# PNG 저장 (Instance Map)
instance_img = Image.fromarray((instance_map * 10).astype(np.uint8))
instance_img.save("instance_map.png")

print("✅ PNG 변환 완료: semantic_map.png, instance_map.png")
