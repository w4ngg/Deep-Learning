import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)        
        self.fc2 = nn.Linear(120, 84)        
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        #conv1 + pool1
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.pool1(x)
        #conv2 + pool2
        x = self.conv2(x)      
        x = self.sigmoid(x)
        x = self.pool2(x)      
        #flatten
        x = x.view(-1, 16 * 5 * 5) 
        #MLP
        x = self.fc1(x)        
        x = self.sigmoid(x)
        x = self.fc2(x)        
        x = self.sigmoid(x)
        x = self.fc3(x)        
        x = self.softmax(x)
        return x

