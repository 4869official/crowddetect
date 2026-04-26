import cv2
import numpy as np
from collections import defaultdict

def extract_clothing_histogram(image, color_space='hsv'):
    """
    提取整张图片的颜色直方图
    :param image: 输入图像
    :param color_space: 颜色空间 ('rgb' 或 'hsv')
    :return: 颜色直方图
    """
    if color_space == 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
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
    # 存储每件衣服的相似性分组
    clothing_groups = defaultdict(list)

    for idx, image_path in enumerate(image_paths):
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            continue

        # 提取整张图片的颜色直方图
        hist = extract_clothing_histogram(image, color_space='hsv')

        # 将直方图存储到列表中
        clothing_histograms.append(hist)

        # 比较当前衣服与其他衣服的相似性
        for i in range(len(clothing_histograms) - 1):
            similarity = compare_histograms(clothing_histograms[-1], clothing_histograms[i])
            if similarity > 0.6:  # 相似性阈值
                clothing_groups[idx].append(i)

    # 统计是否有3个及以上的人穿同一件衣服
    for group_id, group in clothing_groups.items():
        if len(group) >= 3:
            print(f"发现3个及以上的人穿同一件衣服，组ID: {group_id}, 成员: {group}")

if __name__ == "__main__":
    # 图片路径列表
    image_paths = ["C:\\Users\\19025\\Desktop\\clothsimilar\\image1.jpg", "C:\\Users\\19025\\Desktop\\clothsimilar\\image2.jpg", "C:\\Users\\19025\\Desktop\\clothsimilar\\image3.jpg", "C:\\Users\\19025\\Desktop\\clothsimilar\\image4.jpg", "C:\\Users\\19025\\Desktop\\clothsimilar\\image5.jpg"]
    main(image_paths)