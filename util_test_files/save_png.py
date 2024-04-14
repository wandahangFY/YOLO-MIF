import os
import cv2

# 定义输入图像文件夹路径
input_folder = r"G:\wan\data\PVELAD_C\good_corner"

# 定义输出图像文件夹路径
output_folder = r"G:\wan\data\PVELAD_C\good_corner_png"

# 确保输出文件夹存在，如果不存在则创建
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的图像文件
for file_name in os.listdir(input_folder):
    # 构建输入图像文件的完整路径
    input_path = os.path.join(input_folder, file_name)
    if input_path.endswith(".txt"):
        continue
    # 读取灰度图像
    image_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # 构建输出图像文件的完整路径
    output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".png")

    # 将图像保存为16位单通道PNG图像
    cv2.imwrite(output_path, image_gray.astype('uint16')*256)

print("图像保存完成！")
