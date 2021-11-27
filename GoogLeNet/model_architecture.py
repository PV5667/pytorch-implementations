from torch.nn.modules.activation import ReLU
import torchvision
import torch 
import torch.nn as nn
import torch.nn.functional as F

############ output_channels = [1, 3, 5, pool]

class Inception(nn.Module):
    def __init__ (self, input_channels, output_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(input_channels, output_channels[0], kernel_size = 1, stride = (1,1), bias = False)
        self.branch2 = nn.Sequential(nn.Conv2d(input_channels, output_channels[0], kernel_size = 1, stride = (1,1), bias = False), 
                                        nn.ReLU(), nn.Conv2d(output_channels, output_channels[1], kernel_size = 3, stride = (1,1), padding = (1,1), bias = False), 
                                        nn.ReLU())
        self.branch2 = nn.Sequential(nn.Conv2d(input_channels, output_channels[2], kernel_size = 1, stride = (1,1), bias = False), 
                                        nn.ReLU(), nn.Conv2d(output_channels, output_channels[2], kernel_size = 5, stride = (1,1), padding = (1,1), bias = False), 
                                        nn.ReLU()) 