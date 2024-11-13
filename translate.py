import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn.preprocessing import LabelEncoder

# 加载特征和标签数据 
features = pd.read_csv("dataset/features.csv").values 
labels = pd.read_csv("dataset/labels.csv") 
# 提取标签列 
state_labels = labels['state'] # 新鲜度标签 
fruit_labels = labels['fruit_type'] # 水果种类标签

# 使用LabelEncoder将类别标签编码为数值
state_encoder = LabelEncoder()
fruit_encoder = LabelEncoder()
# 将新鲜度标签和水果种类标签编码为数值
state_labels_encoded = state_encoder.fit_transform(state_labels)
fruit_labels_encoded = fruit_encoder.fit_transform(fruit_labels)

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, state_labels_encoded, test_size=0.2, random_state=42)

# 初始化随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
clf.fit(X_train, y_train)
# 使用模型预测测试集
y_pred = clf.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("模型准确率:", accuracy)
# 打印分类报告
print("分类报告:\n", classification_report(y_test, y_pred, target_names=state_encoder.classes_))
# 打印混淆矩阵
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))