import os
from PIL import Image

def resize_images_in_directory(directory, target_size=(128, 128)):
    """
    将指定目录下的所有图像调整为目标大小。

    Args:
        directory (str): 包含图像的目录路径。
        target_size (tuple): 目标图像大小，格式为 (宽度, 高度)。
    """
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(directory, filename)
            try:
                # 打开图像并调整大小
                with Image.open(file_path) as img:
                    img = img.resize(target_size, Image.LANCZOS)  # 使用LANCZOS滤镜
                    img.save(file_path)  # 覆盖原始图像
                print(f"已调整大小: {file_path}")
            except Exception as e:
                print(f"处理 {file_path} 时出错: {e}")

# 使用示例
if __name__ == "__main__":
    directory_path = './val'  # 替换为你的目录路径
    resize_images_in_directory(directory_path)
