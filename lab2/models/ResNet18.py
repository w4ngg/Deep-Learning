import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=21):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2,padding=0)
        self.conv2_1 = ResNetBlock(64, 64, stride=1)
        self.conv2_2 = ResNetBlock(64, 64, stride=1)

        self.conv3_1 = ResNetBlock(64, 128, stride=2)
        self.conv3_2 = ResNetBlock(128, 128, stride=1)

        self.conv4_1 = ResNetBlock(128, 256, stride=2)
        self.conv4_2 = ResNetBlock(256, 256, stride=1)

        self.conv5_1 = ResNetBlock(256, 512, stride=2)
        self.conv5_2 = ResNetBlock(512, 512, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        #block1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        #block2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.maxpool2(x)

        #block3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.maxpool2(x)

        #block4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.maxpool2(x)

        #block5
        x = self.conv5_1(x)
        x = self.conv5_2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
