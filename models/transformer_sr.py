import torch
import torch.nn as nn

class TransformerSR(nn.Module):
    def __init__(self, d_model=64, nhead=2, num_layers=2, patch_size=4, input_dim=4):
        super(TransformerSR, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        # Linear layer to map input channels to d_model
        self.input_projection = nn.Linear(input_dim, d_model)  # Project RGB (3) to d_model (64)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Linear layer to map patches back to image space
        self.fc = nn.Linear(d_model, patch_size * patch_size * input_dim)  # Assuming RGB images

    def forward(self, x):
        # x 的形状应该是 (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()
        # print(x.size())
        # 将图像分割成补丁
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, channels, -1, self.patch_size * self.patch_size)
        x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.d_model)
        # print(x.size())

        # 编码
        encoded = self.encoder(x)  # 输出形状为 (batch_size, height * width / patch_size^2, d_model)
        # print(encoded.size())

        # 将编码后的表示转换回图像空间
        output = self.fc(encoded)  # 形状为 (batch_size, height * width / patch_size^2, patch_size * patch_size * input_dim)
        # print(output.size())

        # 计算输出的形状
        num_patches_height = height // self.patch_size
        num_patches_width = width // self.patch_size
        
        # 将输出调整为图像的形状 (batch_size, channels, height, width)
        output = output.view(batch_size, num_patches_height, num_patches_width, self.patch_size, self.patch_size, channels)
        output = output.permute(0, 5, 1, 3, 2, 4).contiguous()
        output = output.view(batch_size, channels, height, width)

        return output

# 示例用法
if __name__ == "__main__":
    model = TransformerSR()
    input_tensor = torch.randn(2, 3, 128, 128)  # 假设 batch_size=2，RGB 图像，大小为 128x128
    output = model(input_tensor)
    print("Output shape:", output.shape)  # 检查输出形状