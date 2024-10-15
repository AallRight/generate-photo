import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset import ImageDataset
from models.transformer_sr import TransformerSR
from utils.image_utils import blur_image

# 参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 4
num_epochs = 5
learning_rate = 0.001

# 数据集准备
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = ImageDataset(root_dir='./data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 模型
model = TransformerSR().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 训练循环
def train():
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
            outputs = model(blurry_images)
            
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    train()
