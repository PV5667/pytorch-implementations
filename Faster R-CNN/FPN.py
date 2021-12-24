# define the actual structure of the FPN


# using resnet 50 bakckbone, will make a new class called FPN with ResNet50 and add lateral connections, etc.
# To Do: add predictor heads

import torchvision
import torch 
import torch.nn as nn
import torch.nn.functional as F


# We are using ResNet 50 as our backbone for the FPN
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
    #print(x.shape)
    y = F.relu(self.bn1(self.conv1(x)))
    #print(y.shape)
    y = F.relu(self.bn2(self.conv2(y)))
    #print(y.shape)
    y = self.conv3(y)
    #print(y.shape)
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
    #print(x.shape)
    y = F.relu(self.bn1(self.conv1(x)))
    #print(y.shape)
    y = F.relu(self.bn2(self.conv2(y)))
    #print(y.shape)
    y = self.conv3(y)
    #print(y.shape)
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

class Resnet50FPN(nn.Module):
  """ Implements an FPN with ResNet50 as a backbone. Returns the feature maps m5, m4, m3, m2 in that order. """
  def __init__ (self, input_channels, output_channels):
    super().__init__()
    self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride = (2,2), padding = 3)
    self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = (1,1))
    self.conv2 = nn.Sequential(FirstResidualBlock(64, [64, 256]), ResidualBlock(256, [64, 256]), ResidualBlock(256, [64, 256]))
    self.conv3 = nn.Sequential(ResidualBlockwith1x1(256, [128, 512]), ResidualBlock(512, [128, 512]), ResidualBlock(512, [128, 512]), ResidualBlock(512, [128, 512]))
    self.conv4 = nn.Sequential(ResidualBlockwith1x1(512, [256, 1024]), ResidualBlock(1024, [256, 1024]), ResidualBlock(1024, [256, 1024]), ResidualBlock(1024, [256, 1024]), ResidualBlock(1024, [256, 1024]), ResidualBlock(1024, [256, 1024]))
    self.conv5 = nn.Sequential(ResidualBlockwith1x1(1024, [512, 2048]), ResidualBlock(2048, [512, 2048]), ResidualBlock(2048, [512, 2048]))
    self.upsample = nn.Upsample(scale_factor = 2, mode = 'nearest')
    self.conv_c5_1x1 = nn.Conv2d(2048, 256, kernel_size = 1, stride = (1,1), bias = False) 
    self.conv_c4_1x1 = nn.Conv2d(1024, 256, kernel_size = 1, stride = (1,1), bias = False) 
    self.conv_c3_1x1 = nn.Conv2d(512, 256, kernel_size = 1, stride = (1,1), bias = False) 
    self.conv_c2_1x1 = nn.Conv2d(256, 256, kernel_size = 1, stride = (1,1), bias = False) 

  def forward(self, x):
    c1 = F.relu(self.conv1(x))
    print(x.shape)
    c1 = self.pool(c1)
    print(c1.shape)
    c2 = self.conv2(c1)
    print(c2.shape)
    c3 = self.conv3(c2)
    print(c3.shape)
    c4 = self.conv4(c3)
    print(c4.shape)
    c5 = self.conv5(c4)
    print(c5.shape)
    # m5, m4, m3, m2 all must have 256-d because of shared predictor so 1 x 1 conv used
    m5 = self.conv_c5_1x1(c5)
    print("M5: ", m5.shape)
    #print(c5.shape)
    ## layers m4, m3, m2 need to be upsampled by a factor of 2
    m4 = torch.concat((self.upsample(m5), self.conv_c4_1x1(c4)))
    print("M4: ", m4.shape)
    m3 = torch.concat((self.upsample(m4), self.conv_c3_1x1(c3)))
    print("M3: ", m3.shape)
    m2 = torch.concat((self.upsample(m3), self.conv_c2_1x1(c2)))
    print("M2: ", m2.shape) 
    return m5, m4, m3, m2

net = Resnet50FPN(1, 10)
t = torch.randn(1,1, 224, 224)
net.forward(t)
#print(net)

print("done")