import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset import ImageDataset
from utils.image_utils import blur_image
from models.transformer_sr import TransformerSR  # 确保没有循环导入

# 参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 2  # 减小批量大小
num_epochs = 5
learning_rate = 0.001
target_size = (128, 128)  # 目标图像大小
model_save_path = 'transformer_sr_model.pth'  # 模型保存路径

# 数据集准备
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = ImageDataset(root_dir='./data/train', transform=transform, target_size=target_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 使用GradScaler进行混合精度训练
scaler = torch.amp.GradScaler('cuda')

# 训练循环
def train():
    model = TransformerSR().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images in train_loader:
            images = images.to(device)
            
            # 将 Tensor 转换为 PIL 图像
            pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]
            
            # 模糊图像作为输入，清晰图像作为目标
            blurry_images = [blur_image(img) for img in pil_images]  # 调用模糊函数
            
            # 转换回 Tensor
            blurry_images = torch.stack([transforms.ToTensor()(img) for img in blurry_images]).to(device)  # 转换回 Tensor
            
            optimizer.zero_grad()
            
            # 使用autocast进行前向传播
            with torch.amp.autocast('cuda'):
                outputs = model(blurry_images)
                loss = criterion(outputs, images)
            
            # 使用GradScaler进行反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        torch.cuda.empty_cache()  # 清理CUDA缓存

        # 保存模型
        torch.save(model.state_dict(), model_save_path)
        print(f"模型已保存至: {model_save_path}")

if __name__ == "__main__":
    train()