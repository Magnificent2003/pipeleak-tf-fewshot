import torch.nn as nn

# --------------------- Darknet-19 with Two Heads ---------------------
def conv_bn_lrelu(c_in, c_out, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, s, p, bias=False),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(0.1, inplace=True),
    )

class HierarchicalDarknet19(nn.Module):
    def __init__(self, num_classes: int = 4):
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
        self.binary_head = nn.Linear(1024, 2)      # 父类：0=non-leak, 1=leak
        self.child_head  = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        binary_logits = self.binary_head(x)
        child_logits  = self.child_head(x)
        return binary_logits, child_logits