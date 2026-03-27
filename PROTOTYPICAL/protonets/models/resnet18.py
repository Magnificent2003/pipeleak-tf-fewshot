import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        resnet = models.resnet18(pretrained=True)

        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,

            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,

            nn.AdaptiveAvgPool2d((1,1))
        )

    def forward(self, x):

        x = self.encoder(x)

        x = x.view(x.size(0), -1)

        return x