import torch
import torch.nn as nn


def conv_block(in_c, out_c, k=3, s=1, p=1):

    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.1)
    )


class DarkNet19Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            conv_block(3,32),
            nn.MaxPool2d(2),

            conv_block(32,64),
            nn.MaxPool2d(2),

            conv_block(64,128),
            conv_block(128,64,1,1,0),
            conv_block(64,128),
            nn.MaxPool2d(2),

            conv_block(128,256),
            conv_block(256,128,1,1,0),
            conv_block(128,256),
            nn.MaxPool2d(2),

            conv_block(256,512),
            conv_block(512,256,1,1,0),
            conv_block(256,512),
            conv_block(512,256,1,1,0),
            conv_block(256,512),
            nn.AdaptiveAvgPool2d((1,1))
        )

    def forward(self,x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        return x