from torch import nn
from .basic_module import BasicModule


class cnn1(BasicModule):
    """
    一个仿照Alexnet的网络结构
    """
    def __init__(self, num_classes=2500):
        super(cnn1, self).__init__()

        self.model_name = 'cnn1'

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1),  # in_channels, out_channels, kernel_size,stride
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Conv2d(16, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            # nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
        )


        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 2048),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        import ipdb
        #ipdb.set_trace()
        x = self.features(x)
        #import ipdb
        # ipdb.set_trace()
        a=x.shape
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        x = x.view(x.size(0),1, 50, 50)

        return x