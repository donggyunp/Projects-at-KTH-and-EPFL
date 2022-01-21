import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

class self_net(nn.Module):
    def __init__(self, max_kernel):
        super(self_net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, max_kernel, padding=int((max_kernel-1)/2)) 
        self.conv2 = nn.Conv2d(16, 32, max_kernel, padding=int((max_kernel-1)/2))
        self.conv3 = nn.Conv2d(32, 64, max_kernel, padding=int((max_kernel-1)/2))
        self.conv4 = nn.Conv2d(64, 128, max_kernel, padding=int((max_kernel-1)/2))

        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(128*4*4,100,bias=True)
        self.fc2 = nn.Linear(100,10)

    def forward(self, x):
        ## Define forward behavior
        x = F.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.batch_norm2(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.batch_norm3(x)

        x = F.relu(self.conv4(x))
        #x = self.pool(x)
        x = self.batch_norm4(x)

        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
