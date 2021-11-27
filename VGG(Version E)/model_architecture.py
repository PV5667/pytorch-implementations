import torchvision
import torch 
import torch.nn as nn
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__ (self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False)
        ########## maxpool ############
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False)
        self.conv4 = nn.Conv2d(128, 128, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False)
        ########## maxpool ############
        self.conv5 = nn.Conv2d(128, 256, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False)
        self.conv6 = nn.Conv2d(256, 256, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False)
        self.conv7 = nn.Conv2d(256, 256, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False)
        self.conv8 = nn.Conv2d(256, 256, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False)
        ########## maxpool ############
        self.conv9 = nn.Conv2d(256, 512, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False)
        self.conv10 = nn.Conv2d(512, 512, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False)
        self.conv11 = nn.Conv2d(512, 512, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False)
        self.conv12 = nn.Conv2d(512, 512, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False)
        ########## maxpool ############
        self.conv13 = nn.Conv2d(512, 512, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False)
        self.conv14 = nn.Conv2d(512, 512, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False)
        self.conv15 = nn.Conv2d(512, 512, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False)
        self.conv16 = nn.Conv2d(512, 512, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False)
        ########## maxpool ############
        self.classifier = nn.Sequential( nn.Flatten(), nn.Linear(512*14*14, 4096), nn.ReLU(), nn.Linear(4096,4096), nn.ReLU(), nn.Linear(4096,1000))
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.maxpool(x)
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.maxpool(x)
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = self.classifier(x)
        return x

net = VGG16(1, 10)
t = torch.randn(1,1, 224, 224)
net.forward(t)
print(net)
        

