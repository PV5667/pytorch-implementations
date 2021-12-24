import torchvision
import torch 
import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
  def __init__ (self, input_channels, output_channels):
    super().__init__()
    self.conv = nn.Conv2d(input_channels, output_channels, kernel_size = 3, stride=(1, 1), padding = (1,1), bias = False)
    self.bn = nn.BatchNorm2d(output_channels)
  def forward(self, x):
    y = self.conv(x)
    y = self.bn(y)
    y = F.relu(y)
    y = self.conv(y)
    y = self.bn(y)
    y += x
    y = F.relu(y)
    return y

class ResidualBlockwith1x1(nn.Module):
  def __init__ (self, input_channels, output_channels):
    super().__init__()
    self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size = 3, stride = (2, 2), padding = (1,1), bias = False) ###############
    self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size = 3, stride = (1,1),  padding = (1,1), bias = False)
    self.conv1x1 = nn.Conv2d(input_channels, output_channels, kernel_size=(1,1), stride = (2,2),  bias = False) 
    self.bn = nn.BatchNorm2d(output_channels)
  def forward(self, x):
    y = self.conv1(x)
    y = self.bn(y)
    y = F.relu(y)
    y = self.conv2(y)
    y = self.bn(y)
    y = y + self.conv1x1(x) 
    y = F.relu(y)
    return y

class Resnet34(nn.Module):
  def __init__ (self, input_channels, output_channels):
    super().__init__()
    self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride = (2,2), padding = 3)
    self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = (1,1))
    self.conv2 = nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64), ResidualBlock(64, 64))
    self.conv3 = nn.Sequential(ResidualBlockwith1x1(64, 128), ResidualBlock(128, 128), ResidualBlock(128, 128), ResidualBlock(128, 128))
    self.conv4 = nn.Sequential(ResidualBlockwith1x1(128, 256), ResidualBlock(256, 256), ResidualBlock(256, 256), ResidualBlock(256, 256), ResidualBlock(256, 256), ResidualBlock(256, 256))
    self.conv5 = nn.Sequential(ResidualBlockwith1x1(256, 512), ResidualBlock(512, 512), ResidualBlock(512, 512))
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.classifier = nn.Sequential( nn.Flatten(), nn.Linear(512, 10))
  def forward(self, x):
    x = F.relu(self.conv1(x))
    #print(x.shape)
    x = self.pool(x)
    #print(x.shape)
    x = self.conv2(x)
    #print(x.shape)
    x = self.conv3(x)
    #print(x.shape)
    x = self.conv4(x)
    #print(x.shape)
    x = self.conv5(x)
    #print(x.shape)
    x = self.avgpool(x)
    #print(x.shape)
    x = self.classifier(x)
    #print(x.shape)
    return x

net = Resnet34(1, 10)
t = torch.randn(1,1, 224, 224)
net.forward(t)
print(net)
