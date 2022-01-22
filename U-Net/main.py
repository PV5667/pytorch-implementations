import torch
import torchvision                                                                                                                                      
import torch.nn as nn                                                                                                                       
                                                                                                                                            
                                                                                                                                            
                                                                                                                                            

                                                                                                                                            
def UNet_Block(channels):
  """Define the UNet repeating blocks"""
  input_channels = channels[0]
  output_channels = channels[1]
  layers = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size = (3,3), stride = (1,1), padding = (1, 1), bias = False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace = True),
                         nn.Conv2d(output_channels, output_channels, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace = True))
  return layers                                                       
                                                                                                                                            
def conv1x1(input_channels):
  return nn.Sequential(nn.Conv2d(input_channels, input_channels, kernel_size = (1,1), stride = (1, 1), bias = False))
                                                           
                                                                                                                                            
class UNet():                                                                                                                               
  def __init__(self, input_channels, output_channels = 2):                                                                                  
    super().__init__()                                                                                                                      
    self.maxpool = nn.MaxPool2d(kernel_size = (2, 2), stride = 2)
    self.upsample = nn.Upsample(scale_factor = 2, mode = 'nearest')
    self.channels = [
            [input_channels, 64]
            [64, 128],
            [128, 256],
            [256, 512],           
            [512, 1024],
            [1024, 512],
            [512, 256],
            [256, 128],
            [128, 64]
            ]

  def forward(self, x):
    level_1_left = UNet_Block(self.channels[0])(x)
    level_2_left_in = self.maxpool(level_1_left)
    level_2_left = UNet_Block(self.channels[1])(level_2_left_in)
    level_3_left_in = self.maxpool(level_2_left)
    level_3_left = UNet_Block(self.channels[2])(level_3_left_in)
    level_4_left_in = self.maxpool(level_3_left)
    level_4_left = UNet_Block(self.channels[3])(level_4_left_in)
    level_5_in = self.maxpool(level_4_left)
    level_5 = UNet_Block(self.channels[4])(level_5_in)

    level_4_right_in = self.upsample(level_5)
    level_4_right = UNet_Block() 