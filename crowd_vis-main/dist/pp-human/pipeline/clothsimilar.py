import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
from collections import Counter

# 加载YOLO模型用于人体检测
detection_model = YOLO('yolov8n.pt')  # 需安装ultralytics库

# 加载ResNet50模型用于特征提取
feature_extractor = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])  # 移除最后一层
feature_extractor.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_clothes_features(image_paths):
    features = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用YOLO检测人物
        results = detection_model(image_rgb)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for box, cls in zip(boxes, classes):
                if cls == 0:  # 类别0代表'person'
                    x1, y1, x2, y2 = map(int, box[:4])
                    # 裁剪上半部分作为衣服区域（简单启发式方法）
                    height = y2 - y1
                    upper_height = int(height * 0.6)
                    clothes_roi = image_rgb[y1:y1 + upper_height, x1:x2]

                    if clothes_roi.size == 0:
                        continue

                    # 预处理并提取特征
                    try:
                        processed = preprocess(clothes_roi)
                    except:
                        continue
                    with torch.no_grad():
                        feature = feature_extractor(processed.unsqueeze(0))
                    feature_vector = torch.flatten(feature).numpy()
                    features.append(feature_vector)
    return np.array(features)


def analyze_clusters(features, eps=0.35, min_samples=2):
    if len(features) == 0:
        return False
    # 归一化特征向量
    features_normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
    # 使用DBSCAN聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features_normalized)
    counts = Counter(clustering.labels_)
    for label, count in counts.items():
        if label != -1 and count >= 3:  # -1表示噪声点
            return True
    return False


if __name__ == "__main__":
    import sys
    import glob

    if len(sys.argv) < 2:
        print("Usage: python detect_uniform.py <image_directory>")
        sys.exit(1)

    image_dir = sys.argv[1]
    image_paths = glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png")

    features = extract_clothes_features(image_paths)
    has_uniform_group = analyze_clusters(features)

    if has_uniform_group:
        print("检测到至少三人穿着相同服装。")
    else:
        print("未检测到三人或以上穿着相同服装。")