import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.transformer_sr import TransformerSR
from utils.dataset import ImageDataset

# 测试数据集加载
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

test_dataset = ImageDataset(root_dir='./data/val', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerSR().to(device)
model.load_state_dict(torch.load('model.pth'))

# 评估模型
model.eval()
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        # 在这里可以保存或显示输出的图像
