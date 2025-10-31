import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
       
    def forward(self, x):
        id = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += id
        out = F.relu(out)
        return out

class ResNetBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels,out_channels,kernel_size=1)
    def forward(self, x):
        id = self.conv3(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += id
        out = F.relu(out)
        return out