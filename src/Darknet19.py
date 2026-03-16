import torch.nn as nn

# --------------------- Darknet-19 ---------------------
def conv_bn_lrelu(c_in, c_out, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, s, p, bias=False),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(0.1, inplace=True),
    )

class Darknet19(nn.Module):
    """
    简化版 Darknet-19（与原版层次对齐：5个下采样阶段 + 若干 1x1/3x3 交替），
    最后用全局平均池化 + FC 输出到 num_classes。
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_lrelu(3, 32), nn.MaxPool2d(2,2),
            conv_bn_lrelu(32, 64), nn.MaxPool2d(2,2),

            conv_bn_lrelu(64, 128),
            conv_bn_lrelu(128, 64, k=1, s=1, p=0),
            conv_bn_lrelu(64, 128),
            nn.MaxPool2d(2,2),

            conv_bn_lrelu(128, 256),
            conv_bn_lrelu(256, 128, k=1, s=1, p=0),
            conv_bn_lrelu(128, 256),
            nn.MaxPool2d(2,2),

            conv_bn_lrelu(256, 512),
            conv_bn_lrelu(512, 256, k=1, s=1, p=0),
            conv_bn_lrelu(256, 512),
            conv_bn_lrelu(512, 256, k=1, s=1, p=0),
            conv_bn_lrelu(256, 512),
            nn.MaxPool2d(2,2),

            conv_bn_lrelu(512, 1024),
            conv_bn_lrelu(1024, 512, k=1, s=1, p=0),
            conv_bn_lrelu(512, 1024),
            conv_bn_lrelu(1024, 512, k=1, s=1, p=0),
            conv_bn_lrelu(512, 1024),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)
