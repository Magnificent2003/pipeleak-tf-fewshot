import torch.nn as nn
from .encoders import register
from ..modules import *


class DarknetBlock(Module):

    def __init__(self,in_c,out_c):

        super().__init__()

        self.conv=Conv2d(in_c,out_c,3,1,1,bias=False)
        self.bn=BatchNorm2d(out_c)
        self.relu=nn.LeakyReLU(0.1)

    def forward(self,x,params=None,episode=None):

        x=self.conv(x,get_child_dict(params,'conv'))
        x=self.bn(x,get_child_dict(params,'bn'),episode)
        x=self.relu(x)

        return x


class Darknet19(Module):

    def __init__(self):

        super().__init__()

        self.layer1=DarknetBlock(3,32)
        self.pool1=nn.MaxPool2d(2)

        self.layer2=DarknetBlock(32,64)
        self.pool2=nn.MaxPool2d(2)

        self.layer3=DarknetBlock(64,128)
        self.pool3=nn.MaxPool2d(2)

        self.layer4=DarknetBlock(128,256)
        self.pool4=nn.MaxPool2d(2)

        self.layer5=DarknetBlock(256,512)
        self.pool5=nn.MaxPool2d(2)

        self.pool=nn.AdaptiveAvgPool2d(1)

        self.out_dim=512

    def get_out_dim(self):

        return self.out_dim

    def forward(self,x,params=None,episode=None):

        x=self.layer1(x,get_child_dict(params,'layer1'),episode)
        x=self.pool1(x)

        x=self.layer2(x,get_child_dict(params,'layer2'),episode)
        x=self.pool2(x)

        x=self.layer3(x,get_child_dict(params,'layer3'),episode)
        x=self.pool3(x)

        x=self.layer4(x,get_child_dict(params,'layer4'),episode)
        x=self.pool4(x)

        x=self.layer5(x,get_child_dict(params,'layer5'),episode)
        x=self.pool5(x)

        x=self.pool(x).flatten(1)

        return x


@register('darknet19')
def darknet19(**kwargs):
    return Darknet19()