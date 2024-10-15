import torch
import torch.nn as nn

class TransformerSR(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, patch_size=16, input_dim=3):
        super(TransformerSR, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        # Linear layer to map input channels to d_model
        self.input_projection = nn.Linear(input_dim, d_model)  # Project RGB (3) to d_model (256)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Linear layer to map patches back to image space
        self.fc = nn.Linear(d_model, patch_size * patch_size * input_dim)  # Assuming RGB images

    def forward(self, x):
        # x 的形状应该是 (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()

        # 通过局部补丁调整输入形状
        x = x.permute(0, 2, 3, 1)  # (batch_size, height, width, channels)
        x = x.reshape(batch_size, height * width, channels)  # (batch_size, height * width, channels)
    
        # Project input channels to d_model
        x = self.input_projection(x)  # 形状变为 (batch_size, height * width, d_model)

        # 通过 Transformer 编码器
        encoded = self.encoder(x)  # 输出形状为 (batch_size, height * width, d_model)

        # 将编码后的表示转换回图像空间
        output = self.fc(encoded)  # 形状为 (batch_size, height * width, patch_size * patch_size * input_dim)
        
        # 将输出调整为图像的形状 (batch_size, input_dim, new_height, new_width)
        output = output.reshape(batch_size, -1, self.patch_size, self.patch_size)  # 注意这里的 -1 会自动推导出高度
        
        return output.permute(0, 2, 3, 1)  # 最后调整为 (batch_size, new_height, new_width, input_dim)

# 示例用法
if __name__ == "__main__":
    model = TransformerSR()
    # 创建一个假输入，形状为 (batch_size, channels, height, width)
    input_tensor = torch.randn(8, 3, 64, 64)  # 假设 batch_size=8，RGB 图像，大小为 64x64
    output = model(input_tensor)
    print("Output shape:", output.shape)  # 检查输出形状
    # Output shape: torch.Size([8, 64, 64, 3])