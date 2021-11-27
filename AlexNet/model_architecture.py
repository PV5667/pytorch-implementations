import torchvision.datasets as datasets
import torch
import torchvision 
import torch.nn.functional as F 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms  
from torch import optim 
from torch import nn  
from torch.utils.data import DataLoader 
from tqdm import tqdm

class AlexNet(nn.Module):
  def __init__ (self, input_channels, output_channels):
    super().__init__()
    self.conv1 = nn.Conv2d(input_channels, 96, kernel_size = 11, stride = 4, bias = False)
    self.max_pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2)
    self.conv2 = nn.Conv2d(96, 256, kernel_size = 5, stride = (1,1), padding=(2,2))
    self.max_pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2)
    self.conv3 = nn.Conv2d(256, 384, kernel_size = 3, stride =(1,1), padding = (1,1))
    self.conv4 = nn.Conv2d(384, 384, kernel_size = 3, stride =(1,1), padding = (1,1))
    self.conv5 = nn.Conv2d(384, 256, kernel_size = 3, stride =(1,1), padding = (1,1))
    self.max_pool3 = nn.MaxPool2d(kernel_size = 3, stride = 2)
    self.dropout = nn.Dropout2d(0.5)
    self.classifier = nn.Sequential( nn.Flatten(), nn.Linear(256*5*5, 10))
  def forward(self, x):
    x = F.relu(self.conv1(x))
    print(x.shape)
    x = self.max_pool1(x)
    print(x.shape)
    x = F.relu(self.conv2(x))
    print(x.shape)
    x = self.max_pool2(x)
    print(x.shape)
    x = F.relu(self.conv3(x))
    print(x.shape)
    x = F.relu(self.conv4(x))
    print(x.shape)
    x = F.relu(self.conv5(x))
    print(x.shape)
    x = self.max_pool3(x)
    print(x.shape)
    x = self.dropout(x)
    print(x.shape)
    x = self.classifier(x)
    print(x.shape)
    return x

net = AlexNet(1, 10)
t = torch.randn(1,1, 224, 224)
net.forward(t)
print(net)