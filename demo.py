import torch
from torchvision import transforms
from PIL import Image
import os
from models.transformer_sr import TransformerSR  # 导入你的 Transformer 模型

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 根据需要调整图像大小
    transforms.ToTensor(),  # 转换为 Tensor
])

def load_model(model_path, device):
    # 加载训练好的模型
    model = TransformerSR().to(device)  # 根据你的模型定义来实例化
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式
    return model

def predict(model, image_path, device):
    # 加载和转换输入图片
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(input_image).unsqueeze(0).to(device)  # 增加 batch 维度并移动到设备

    with torch.no_grad():  # 关闭梯度计算
        output_tensor = model(input_tensor)  # 模型推理

    output_image = output_tensor.squeeze(0).permute(1, 2, 0).cpu()  # 从 (C, H, W) 转换为 (H, W, C) 并移动到 CPU
    output_image = (output_image.clamp(0, 1) * 255).numpy().astype("uint8")  # 处理输出值
    return Image.fromarray(output_image)

def main():
    # 指定模型路径和输入图片路径
    model_path = 'transformer_sr_model.pth'  # 替换为你的模型文件路径
    input_image_path = '55779.jpg'  # 替换为你的模糊图片路径
    output_image_path = 'output_image.png'  # 输出清晰图片的路径

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    clear_image = predict(model, input_image_path, device)
    clear_image.save(output_image_path)  # 保存生成的清晰图片
    print(f"清晰图片已保存至: {output_image_path}")

if __name__ == "__main__":
    main()