import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# 初始化YOLOv8模型
model = YOLO("yolov8n.pt")  # 使用预训练的YOLOv8模型

def extract_clothing_histogram(image, bbox, color_space='hsv'):
    """
    提取衣服区域的颜色直方图
    :param image: 输入图像
    :param bbox: 人物的边界框 (x, y, w, h)
    :param color_space: 颜色空间 ('rgb' 或 'hsv')
    :return: 颜色直方图
    """
    x, y, w, h = bbox
    clothing_roi = image[y:y+h, x:x+w]

    if color_space == 'hsv':
        clothing_roi = cv2.cvtColor(clothing_roi, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([clothing_roi], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    """
    比较两个颜色直方图的相似性
    :param hist1: 第一个直方图
    :param hist2: 第二个直方图
    :param method: 比较方法 (默认使用相关系数)
    :return: 相似性得分
    """
    return cv2.compareHist(hist1, hist2, method)

def main(image_paths):
    # 存储每件衣服的直方图
    clothing_histograms = []
    # 存储每件衣服的边界框
    clothing_bboxes = []
    # 存储每件衣服的相似性分组
    clothing_groups = defaultdict(list)

    for idx, image_path in enumerate(image_paths):
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            continue

        # 使用YOLOv8检测图片中的人
        results = model(image)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = box.cls[0].item()
                if cls == 0:  # 0 表示 "person" 类
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    w = x2 - x1
                    h = y2 - y1
                    bbox = (x1, y1, w, h)

                    # 提取衣服区域的颜色直方图
                    hist = extract_clothing_histogram(image, bbox, color_space='hsv')

                    # 将直方图和边界框存储到列表中
                    clothing_histograms.append(hist)
                    clothing_bboxes.append(bbox)

                    # 比较当前衣服与其他衣服的相似性
                    for i in range(len(clothing_histograms) - 1):
                        similarity = compare_histograms(clothing_histograms[-1], clothing_histograms[i])
                        if similarity > 0.8:  # 相似性阈值
                            clothing_groups[idx].append(i)

    # 统计是否有3个及以上的人穿同一件衣服
    for group_id, group in clothing_groups.items():
        if len(group) >= 2:
            print(f"发现3个及以上的人穿同一件衣服，组ID: {group_id}, 成员: {group}")

if __name__ == "__main__":
    # 图片路径列表
    image_paths = ["C:\\Users\\19025\\Desktop\\clothsimilar\\image1.jpg", "C:\\Users\\19025\\Desktop\\clothsimilar\\image2.jpg", "C:\\Users\\19025\\Desktop\\clothsimilar\\image3.jpg"]
    main(image_paths)