import torch
import torch.nn as nn
import torch.optim as optim

# Transformer 模型定义
class TransformerModel(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super(TransformerModel, self).__init__()
        # Transformer 编码器
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), 
            num_layers=num_layers
        )
        # 图像生成的线性层
        self.fc = nn.Linear(d_model, 256*256*3)  # 假设生成 256x256 RGB 图像

    def forward(self, x):
        encoded = self.encoder(x)
        out = self.fc(encoded)
        # 将输出转换为图像尺寸
        return out.view(-1, 3, 256, 256)

# 创建模型
model = TransformerModel()

# 假设有一个输入张量
input_tensor = torch.randn(32, 10, 256)  # [batch_size, seq_length, d_model]

# 运行模型，生成图像
output = model(input_tensor)

print(output.shape)  # 输出应该是 [32, 3, 256, 256]，表示 32 张 256x256 的 RGB 图像
