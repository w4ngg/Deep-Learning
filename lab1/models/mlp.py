import torch
import torch.nn as nn


class MLP1(nn.Module):
    
    def __init__(self,input_size,output_size,num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size,output_size)
        self.softmax=nn.Softmax(dim=1)
        self.classifier = nn.Linear(output_size,num_classes)
        self.flatten = nn.Flatten()
    def forward(self,x):
        x = self.flatten(x)
        out = self.fc1(x)
        out = self.classifier(out)
        out = self.softmax(out)
        return out
    
class MLP2(nn.Module):
    
    def __init__(self,input_size,hidden1,hidden2,output_size,num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden1)
        self.fc2 = nn.Linear(hidden1,hidden2)
        self.fc3 = nn.Linear(hidden2,output_size)
        self.softmax=nn.Softmax(dim=1)
        self.classifier = nn.Linear(output_size,num_classes)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
    def forward(self,x):
        x = self.flatten(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.classifier(out)
        out = self.softmax(out)
        return out