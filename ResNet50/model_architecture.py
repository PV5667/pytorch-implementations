import torchvision
import torch 
import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
  def __init__ (self, input_channels, output_channels):
    super().__init__()
    self.conv1 = nn.Conv2d(input_channels, output_channels[0], kernel_size = 1, stride = (1,1), bias = False)
    self.conv2 = nn.Conv2d(output_channels[0], output_channels[0], kernel_size = 3, stride=(1, 1), padding = (1,1), bias = False)
    self.conv3 = nn.Conv2d(output_channels[0], output_channels[1], kernel_size = 1, stride = (1,1), bias = False)
    self.bn1 = nn.BatchNorm2d(output_channels[0])
    self.bn2 = nn.BatchNorm2d(output_channels[0])
    self.bn3 = nn.BatchNorm2d(output_channels[1])
  def forward(self, x):
    y = F.relu(self.bn1(self.conv1(x)))
    y = F.relu(self.bn2(self.conv2(y)))
    y = self.conv3(y)
    y = self.bn3(y)
    y += x
    y = F.relu(y)
    return y


# the first residual block when going from C! to C2. Helps with the initial dimensionality mismatch while not downsampling the image
# This falls into the category "B" in the ResNet Paper
class FirstResidualBlock(nn.Module):
  def __init__ (self, input_channels, output_channels):
    super().__init__()
    self.conv1 = nn.Conv2d(input_channels, output_channels[0], kernel_size = 1, stride = (1,1), bias = False)
    self.conv2 = nn.Conv2d(output_channels[0], output_channels[0], kernel_size = 3, stride=(1, 1), padding = (1,1), bias = False)
    self.conv3 = nn.Conv2d(output_channels[0], output_channels[1], kernel_size = 1, stride = (1,1), bias = False)
    self.bn1 = nn.BatchNorm2d(output_channels[0])
    self.bn2 = nn.BatchNorm2d(output_channels[0])
    self.bn3 = nn.BatchNorm2d(output_channels[1])
    self.conv1x1 = nn.Conv2d(input_channels, output_channels[1], kernel_size = (1,1), stride = (1,1), bias = False)
  def forward(self, x):
    y = F.relu(self.bn1(self.conv1(x)))
    y = F.relu(self.bn2(self.conv2(y)))
    y = self.conv3(y)
    y = self.bn3(y)
    y = y + self.conv1x1(x)
    y = F.relu(y)
    return y

class ResidualBlockwith1x1(nn.Module):
  def __init__ (self, input_channels, output_channels):
    super().__init__()
    self.conv1 = nn.Conv2d(input_channels, output_channels[0], kernel_size = 1, stride = (2,2), bias = False)
    self.conv2 = nn.Conv2d(output_channels[0], output_channels[0], kernel_size = 3, stride=(1, 1), padding = (1,1), bias = False)
    self.conv3 = nn.Conv2d(output_channels[0], output_channels[1], kernel_size = 1, stride = (1,1), bias = False)
    self.conv1x1 = nn.Conv2d(input_channels, output_channels[1], kernel_size=(1,1), stride = (2,2),  bias = False) ############### stays same without regard to the architecture??
    self.bn1 = nn.BatchNorm2d(output_channels[0])
    self.bn2 = nn.BatchNorm2d(output_channels[0])
    self.bn3 = nn.BatchNorm2d(output_channels[1])
  def forward(self, x):
    y = F.relu(self.bn1(self.conv1(x)))
    y = F.relu(self.bn2(self.conv2(y)))
    y = self.bn3(self.conv3(y))
    y = y + self.conv1x1(x) 
    y = F.relu(y)
    return y

class Resnet50(nn.Module):
  def __init__ (self, input_channels, output_channels):
    super().__init__()
    self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride = (2,2), padding = 3)
    self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = (1,1))
    self.conv2 = nn.Sequential(FirstResidualBlock(64, [64, 256]), ResidualBlock(256, [64, 256]), ResidualBlock(256, [64, 256]))
    self.conv3 = nn.Sequential(ResidualBlockwith1x1(256, [128, 512]), ResidualBlock(512, [128, 512]), ResidualBlock(512, [128, 512]), ResidualBlock(512, [128, 512]))
    self.conv4 = nn.Sequential(ResidualBlockwith1x1(512, [256, 1024]), ResidualBlock(1024, [256, 1024]), ResidualBlock(1024, [256, 1024]), ResidualBlock(1024, [256, 1024]), ResidualBlock(1024, [256, 1024]), ResidualBlock(1024, [256, 1024]))
    self.conv5 = nn.Sequential(ResidualBlockwith1x1(1024, [512, 2048]), ResidualBlock(2048, [512, 2048]), ResidualBlock(2048, [512, 2048]))
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.classifier = nn.Sequential( nn.Flatten(), nn.Linear(2048, 10))
  def forward(self, x):
    x = F.relu(self.conv1(x))
    print(x.shape)
    x = self.pool(x)
    print(x.shape)
    x = self.conv2(x)
    print(x.shape)
    x = self.conv3(x)
    print(x.shape)
    x = self.conv4(x)
    print(x.shape)
    x = self.conv5(x)
    print(x.shape)
    x = self.avgpool(x)
    x = self.classifier(x)
    return x

net = Resnet50(1, 10)
t = torch.randn(1,1, 224, 224)
net.forward(t)
print(net)