import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. 定义数据集加载器 (使用CIFAR-10作为示例)
class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, _ = self.data[idx]  # 我们只需要图像，不需要标签
        if self.transform:
            image = self.transform(image)
        return image

# 2. 将图像划分为块，并转化为序列 (例如每个图像为 256x256)
def img_to_patches(img, patch_size=16):
    # 将图像分为 [num_patches, patch_size*patch_size*channels]
    patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(img.size(0), -1, patch_size * patch_size * 3)
    return patches

# 3. Transformer 模型定义
class TransformerModel(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, patch_size=16):
        super(TransformerModel, self).__init__()
        self.patch_size = patch_size
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), 
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, patch_size * patch_size * 3)  # 每个patch的输出

    def forward(self, x):
        # 对图像块进行 Transformer 编码
        encoded = self.encoder(x)
        out = self.fc(encoded)
        return out

# 4. 重建图像函数（从块还原为图像）
def patches_to_img(patches, img_size=256, patch_size=16):
    num_patches = (img_size // patch_size) ** 2
    patches = patches.view(-1, 3, patch_size, patch_size)
    img = patches.permute(0, 2, 3, 1).contiguous()
    img = img.view(-1, 3, img_size, img_size)
    return img

# 5. 模型训练设置
def train(model, dataloader, optimizer, criterion, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for imgs in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            
            # 将图像分块
            patches = img_to_patches(imgs)
            
            # 使用 Transformer 生成图像
            outputs = model(patches)
            
            # 还原图像并计算损失
            outputs = patches_to_img(outputs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')

# 6. 主函数
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # 使用CIFAR-10数据集作为例子
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_dataset = ImageDataset(train_data, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 定义模型、损失函数和优化器
    model = TransformerModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    train(model, train_loader, optimizer, criterion, num_epochs=5)

    # 测试并显示图像
    dataiter = iter(train_loader)
    images = dataiter.next().to(device)
    with torch.no_grad():
        patches = img_to_patches(images)
        outputs = model(patches)
        output_imgs = patches_to_img(outputs)

    # 显示原始图像和生成的图像
    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    for i in range(4):
        axs[0, i].imshow(images[i].permute(1, 2, 0).cpu().numpy())
        axs[0, i].set_title('Original')
        axs[1, i].imshow(output_imgs[i].permute(1, 2, 0).cpu().numpy())
        axs[1, i].set_title('Generated')
    plt.show()
