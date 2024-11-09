import cv2
import numpy as np

## 说明：
## 剩下的部分为model学习部分，可以使用标签lable[苹果、橙子、香蕉], 然后再根据提取出的features进行分类。这样便能对应上不同水果
## 同理，在套入训练后的model，也能根据features判断出水果的好坏。
## 接下来的同学加油！！

## 需要做：
## 1. 预处理部分：对图像进行预处理，包括调整尺寸、高斯模糊、图像增强、颜色空间转换、图像分割、边缘检测等。
## 2. 特征提取部分：提取图像的颜色和形状特征，组合成一个特征向量。
## 3. 训练模型：使用训练集训练模型，并保存模型。
## 4. 测试模型：使用测试集测试模型，并计算准确率。
## 5. 部署模型：将训练好的模型部署到生产环境，并接收图像输入，输出预测结果。


##目前只差训练两步，叶子应该是model已经训练好了，只差函数的调试
##组长在接受图像，摄像头部分的调试未知进度，大家加油！！！


# 1. 预处理部分
def resize_image(image, size=(512,512)):
    return cv2.resize(image, size)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def enhance_image(image):
    # 转换到YUV颜色空间以增强亮度
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    enhanced_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    return enhanced_image

def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def segment_image(image_hsv):
    # 设定颜色阈值，提取符合颜色的区域
    lower_bound = np.array([10, 50, 50])  # 自定义下限
    upper_bound = np.array([30, 255, 255])  # 自定义上限
    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
    segmented_img = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)
    return segmented_img

def detect_edges(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

def preprocess_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    # 调整尺寸
    image = resize_image(image)
    # 高斯模糊
    image = apply_gaussian_blur(image)
    # 图像增强
    image = enhance_image(image)
    # 转换到HSV
    image_hsv = convert_to_hsv(image)
    # 图像分割
    segmented_image = segment_image(image_hsv)
    # 边缘检测
    edges = detect_edges(cv2.cvtColor(segmented_image, cv2.COLOR_HSV2BGR))
    return segmented_image, edges

# 2. 特征提取部分
# 提取颜色直方图特征
def extract_color_features(image_hsv):
    """
    计算图像的颜色直方图特征。
    将图像的HSV直方图进行归一化，并展平成一维特征向量。
    """
    hist = cv2.calcHist([image_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # 归一化并展平成一维数组
    return hist

# 形状特征提取函数
def extract_shape_features(image, fruit_type):
    """
    根据水果类型提取适合的形状特征。
    对于香蕉，提取长宽比和方向；对于其他水果，提取面积、周长和圆度。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        contour = max(contours, key=cv2.contourArea)  # 选择最大轮廓
        
        if fruit_type == "banana":
            # 对香蕉提取长宽比和方向角度
            aspect_ratio = calculate_aspect_ratio(contour)
            orientation = calculate_orientation(contour)
            return [aspect_ratio, orientation]
        
        else:
            # 对其他水果提取面积、周长和圆度
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            roundness = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter != 0 else 0
            return [area, perimeter, roundness]
    else:
        # 如果没有找到轮廓，返回默认值
        return [0, 0, 0] if fruit_type != "banana" else [0, 0]

# 辅助函数：计算长宽比
def calculate_aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return float(w) / h if h != 0 else 0

# 辅助函数：计算旋转角度
def calculate_orientation(contour):
    rect = cv2.minAreaRect(contour)
    return rect[2]

# 综合特征提取函数
def extract_features(image, fruit_type):
    """
    提取图像的颜色和形状特征，组合成一个特征向量。
    """
    color_features = extract_color_features(image)
    shape_features = extract_shape_features(image, fruit_type)
    return np.concatenate([color_features, shape_features])  # 将颜色和形状特征组合成一个特征向量

# 判断水果是否腐烂
def detect_rot(img_hsv):
    lower_rot_bound = np.array([0, 100, 50])  # 腐烂颜色下限
    upper_rot_bound = np.array([10, 255, 150])  # 腐烂颜色上限
    mask = cv2.inRange(img_hsv, lower_rot_bound, upper_rot_bound)
    rot_area = cv2.countNonZero(mask)
    total_area = img_hsv.shape[0] * img_hsv.shape[1]
    rot_percentage = (rot_area / total_area) * 100
    return rot_percentage > 10  # 设置腐烂的阈值

# 判断水果是否成熟
def detect_ripeness(img_hsv, fruit_type):
    # 设置不同水果的颜色范围，根据成熟状态调整
    if fruit_type == "Apple":
        # 例如：成熟的红苹果为红色，未成熟为绿色
        ripe_lower = np.array([0, 50, 50])     # 成熟颜色下限（红色）
        ripe_upper = np.array([10, 255, 255])  # 成熟颜色上限
        unripe_lower = np.array([35, 50, 50])  # 未成熟颜色下限（绿色）
        unripe_upper = np.array([85, 255, 255])  # 未成熟颜色上限
    elif fruit_type == "Orange":
        # 橙子的成熟色为橙黄色
        ripe_lower = np.array([10, 150, 150])  # 成熟颜色下限（橙色）
        ripe_upper = np.array([25, 255, 255])  # 成熟颜色上限
        unripe_lower = np.array([35, 50, 50])  # 未成熟颜色下限（绿色）
        unripe_upper = np.array([85, 255, 255])  # 未成熟颜色上限
    elif fruit_type == "Banana":
        # 成熟香蕉为黄色
        ripe_lower = np.array([15, 60, 60])  # 成熟颜色下限（黄色）
        ripe_upper = np.array([40, 255, 255])  # 成熟颜色上限
        unripe_lower = np.array([35, 50, 50])  # 未成熟颜色下限（绿色）
        unripe_upper = np.array([85, 255, 255])  # 未成熟颜色上限

    # 创建掩码，判断图像中成熟和未成熟区域的占比
    ripe_mask = cv2.inRange(img_hsv, ripe_lower, ripe_upper)
    unripe_mask = cv2.inRange(img_hsv, unripe_lower, unripe_upper)
    
    # 计算成熟区域和未成熟区域的像素数
    ripe_area = cv2.countNonZero(ripe_mask)
    unripe_area = cv2.countNonZero(unripe_mask)

    print(ripe_mask, unripe_mask)
    print(ripe_area, unripe_area)
    
    # 判断成熟度：如果成熟区域较大，返回 True 表示成熟；否则返回 False 表示未成熟
    if ripe_area > unripe_area:
        return True  # 成熟
    else:
        return False  # 未成熟
    
   



if __name__ == '__main__':
    # 读取图像
    image_path = "D:\VS Code\Coding\CV\Group - Assignment\Orange_Notgood.jpg" # 请根据自己的图像路径替换
    segmented_image, edges = preprocess_image(image_path)
    image_hsv = cv2.cvtColor(segmented_image, cv2.COLOR_HSV2BGR)
    #image_hsv = cv2.imread(image_path,cv2.COLOR_HSV2BGR)
    is_rot = detect_rot(segmented_image)  # 检测腐烂
    is_ripe = detect_ripeness(segmented_image, "Orange")  # 检测成熟  模型训练还没自动检测时，先手动输入
    if is_ripe:
        print("水果成熟")
    else:
        print("水果未成熟")
    if is_rot:
        print("水果腐烂")
    else:
        print("水果未腐烂")
    # 显示图像
    #cv2.imshow("segmented_img", segmented_image)
    #cv2.imshow("edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

