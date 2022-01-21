import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

class net_8(nn.Module):
    def __init__(self, max_kernel):
        super(net_8, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, max_kernel, padding=int((max_kernel-1)/2)) 
        self.conv2 = nn.Conv2d(6, 12, max_kernel, padding=int((max_kernel-1)/2))
        self.conv3 = nn.Conv2d(12, 24, max_kernel, padding=int((max_kernel-1)/2))
        self.conv4 = nn.Conv2d(24, 36, max_kernel, padding=int((max_kernel-1)/2))
        self.conv5 = nn.Conv2d(36, 48, max_kernel, padding=int((max_kernel-1)/2))
        self.conv6 = nn.Conv2d(48, 60, max_kernel, padding=int((max_kernel-1)/2))
        self.conv7 = nn.Conv2d(60, 72, max_kernel, padding=int((max_kernel-1)/2))
        self.conv8 = nn.Conv2d(72, 84, max_kernel, padding=int((max_kernel-1)/2))

        self.batch_norm1 = nn.BatchNorm2d(6)
        self.batch_norm2 = nn.BatchNorm2d(12)
        self.batch_norm3 = nn.BatchNorm2d(24)
        self.batch_norm4 = nn.BatchNorm2d(36)
        self.batch_norm5 = nn.BatchNorm2d(48)
        self.batch_norm6 = nn.BatchNorm2d(60)
        self.batch_norm7 = nn.BatchNorm2d(72)
        self.batch_norm8 = nn.BatchNorm2d(84)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(84*8*8,500,bias=True)
        self.fc2 = nn.Linear(500,10)

    def forward(self, x):
        ## Define forward behavior
        x = F.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.batch_norm2(x)

        x = F.relu(self.conv3(x))
        #x = self.pool(x)
        x = self.batch_norm3(x)

        x = F.relu(self.conv4(x))
        #x = self.pool(x)
        x = self.batch_norm4(x)

        x = F.relu(self.conv5(x))
        #x = self.pool(x)
        x = self.batch_norm5(x)

        x = F.relu(self.conv6(x))
        #x = self.pool(x)
        x = self.batch_norm6(x)

        x = F.relu(self.conv7(x))
        #x = self.pool(x)
        x = self.batch_norm7(x)

        x = F.relu(self.conv8(x))
        #x = self.pool(x)
        x = self.batch_norm8(x)

        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class net_flex(nn.Module):
    def __init__(self):
        super(net_flex, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, max_kernel, padding=int((max_kernel-1)/2)) 
        self.conv2 = nn.Conv2d(6, 12, max_kernel, padding=int((max_kernel-1)/2))
        self.conv3 = nn.Conv2d(12, 24, max_kernel, padding=int((max_kernel-1)/2))
        self.conv4 = nn.Conv2d(24, 36, max_kernel, padding=int((max_kernel-1)/2))
        self.conv5 = nn.Conv2d(36, 48, max_kernel, padding=int((max_kernel-1)/2))
        self.conv6 = nn.Conv2d(48, 60, max_kernel, padding=int((max_kernel-1)/2))
        self.conv7 = nn.Conv2d(60, 72, max_kernel, padding=int((max_kernel-1)/2))
        self.conv8 = nn.Conv2d(72, 84, max_kernel, padding=int((max_kernel-1)/2))

        self.batch_norm1 = nn.BatchNorm2d(6)
        self.batch_norm2 = nn.BatchNorm2d(12)
        self.batch_norm3 = nn.BatchNorm2d(24)
        self.batch_norm4 = nn.BatchNorm2d(36)
        self.batch_norm5 = nn.BatchNorm2d(48)
        self.batch_norm6 = nn.BatchNorm2d(60)
        self.batch_norm7 = nn.BatchNorm2d(72)
        self.batch_norm8 = nn.BatchNorm2d(84)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(84*8*8,500,bias=True)
        self.fc2 = nn.Linear(500,10)

    def forward(self, x):
        ## Define forward behavior
        x = F.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.batch_norm2(x)

        x = F.relu(self.conv3(x))
        #x = self.pool(x)
        x = self.batch_norm3(x)

        x = F.relu(self.conv4(x))
        #x = self.pool(x)
        x = self.batch_norm4(x)

        x = F.relu(self.conv5(x))
        #x = self.pool(x)
        x = self.batch_norm5(x)

        x = F.relu(self.conv6(x))
        #x = self.pool(x)
        x = self.batch_norm6(x)

        x = F.relu(self.conv7(x))
        #x = self.pool(x)
        x = self.batch_norm7(x)

        x = F.relu(self.conv8(x))
        #x = self.pool(x)
        x = self.batch_norm8(x)

        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
