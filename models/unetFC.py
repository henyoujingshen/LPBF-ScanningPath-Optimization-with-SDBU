from torch import nn
from .basic_module import BasicModule
from .AttentionModule import AttentionModule
from .unet_parts import *

##消融实验的unet变体
class unet_fc(nn.Module):
    def __init__(self, bilinear=False):
        super(unet_fc, self).__init__()
        self.model_name = 'unet_fc'
        n_channels = 1
        n_classes = 1
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 保留原始的下采样层
        self.shared_layers = nn.Sequential(
            DoubleConv(n_channels, 32),
            Down(32, 64),
            Down(64, 128)
        )

        # 替换上采样层为全连接层
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 1000),  # 尺寸调整为与13x13的特征图相匹配
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, n_classes * 50 * 50),  # 输出尺寸调整为50x50
            nn.ReLU()
        )

    def forward(self, x):
        x = self.shared_layers(x)
        x = self.fc(x)
        x = x.view(-1, self.n_classes, 50, 50)  # 调整输出形状以匹配图像尺寸
        return x
