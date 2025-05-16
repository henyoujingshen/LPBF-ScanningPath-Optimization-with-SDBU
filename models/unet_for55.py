from torch import nn
from .basic_module import BasicModule
from .AttentionModule import AttentionModule
from .unet_parts import *
##unet for
class unet_for55(BasicModule):
    def __init__(self,  bilinear=False):
        super(unet_for55, self).__init__()
        self.model_name = 'unet_for55'
        n_channels=1
        n_classes=1
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 减少初始通道数
        self.inc = DoubleConv(n_channels, 32)

        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        logits = self.outc(x1)
        print(logits.shape)
        return logits