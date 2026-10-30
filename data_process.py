import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 定义输入和输出路径
# 为了演示，我们将使用描述性的文件名，并确保在脚本运行目录下保存它们
input_image_path = r'D:\seg_data\Dataset_organoids\image1.png'  # 使用原始字符串前缀 r'' 修正路径问题

# 1. 加载图像（加载为 BGR）
image = cv2.imread(input_image_path)

# 检查图像是否加载成功
if image is None:
    print(f"Error: Could not load image from {input_image_path}.")
    print("Please ensure the path is correct.")
    print("Creating a sample placeholder image for demonstration purposes...")

    # 替代方案：生成一个占位符图像以演示保存逻辑
    # 绿色正方形、红色正方形、蓝色正方形
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[50:150, 50:150] = [0, 255, 0]  # 绿色
    image[150:, 150:] = [0, 0, 255]  # 红色
    image[:50, :50] = [255, 0, 0]  # 蓝色

    # 定义占位符文件名
    output_rotated_path = 'D:\seg_data\Dataset_organoids\placeholder_rotated.png'
    output_flipped_path = 'D:\seg_data\Dataset_organoids\placeholder_flipped.png'
    output_cropped_path = 'D:\seg_data\Dataset_organoids\placeholder_cropped.png'
    output_brightened_path = 'D:\seg_data\Dataset_organoids\placeholder_brightened.png'
    print("Using placeholder names for demonstration.")
else:
    # 2. 如果加载成功，定义输出文件名（在脚本运行目录下保存）
    # 使用描述性文件名以区分不同的增强操作
    output_rotated_path = 'D:\seg_data\Dataset_organoids\placeholder_rotated.png'
    output_flipped_path = 'D:\seg_data\Dataset_organoids\placeholder_flipped.png'
    output_cropped_path = 'D:\seg_data\Dataset_organoids\placeholder_cropped.png'
    output_brightened_path = 'D:\seg_data\Dataset_organoids\placeholder_brightened.png'
    print(f"Image loaded successfully from {input_image_path}")

# 2. 增强操作 (保持 BGR 格式，不进行 RGB 转换)

# --- 操作 1: Rotate（旋转，保持 BGR） ---
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
# 顺时针旋转 90 度
M_rotate = cv2.getRotationMatrix2D(center, -90, 1.0)
rotated_image = cv2.warpAffine(image, M_rotate, (w, h))

# --- 操作 2: Flip（翻转，保持 BGR） ---
flipped_image = cv2.flip(image, 1)

# --- 操作 3: Crop（裁剪，保持 BGR） ---
# 定义裁剪区域：[起始y:结束y, 起始x:结束x]
# 确保裁剪区域在占位符图像（200x200）范围内
cropped_image = image[50:150, 50:150]

# --- 操作 4: Brightness changes（亮度变化，保持 BGR） ---
# alpha = 1.0 (无亮度变化)，alpha > 1.0 (增加亮度)，alpha < 1.0 (减少亮度)
brightened_image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

# 3. 单独保存处理后的图像
# 注意：cv2.imwrite 预期 BGR 数组，这正是我们处理后的图像
cv2.imwrite(output_rotated_path, rotated_image)
cv2.imwrite(output_flipped_path, flipped_image)
cv2.imwrite(output_cropped_path, cropped_image)
cv2.imwrite(output_brightened_path, brightened_image)

# 如果需要保存原始图像，也可以添加以下行（通常不需要）
# cv2.imwrite('original.png', image)

# 打印消息确认保存
print("\nEnhanced images saved successfully in the current directory:")
print(f"- {output_rotated_path}")
print(f"- {output_flipped_path}")
print(f"- {output_cropped_path}")
print(f"- {output_brightened_path}")