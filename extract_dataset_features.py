import cv2
import pandas as pd
import numpy as np
from feature_extraction import extract_features

# 读取数据集标签
data = pd.read_csv("dataset/dataset_labels.csv")

# 定义保存特征的列表
features_list = []
labels_list = []

# 遍历数据集
for index, row in data.iterrows():
    image_path = row['image_path']
    state = row['state']  # 新鲜度标签
    fruit_type = row['fruit_type']  # 水果种类
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        continue
    
    # 提取特征
    features = extract_features(image, fruit_type)
    features_list.append(features)
    
    # 将标签组合为一组
    labels_list.append([state, fruit_type])

# 转换为DataFrame并保存特征和标签
features_df = pd.DataFrame(features_list)
labels_df = pd.DataFrame(labels_list, columns=["state", "fruit_type"])

# 保存到文件
features_df.to_csv("dataset/features.csv", index=False)
labels_df.to_csv("dataset/labels.csv", index=False)

print("特征和标签已保存到 features.csv 和 labels.csv")
