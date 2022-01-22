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
                                                                                                                                            
def conv1x1(input_channels, output_channels):
  return nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size = (1,1), stride = (1, 1), bias = False))
                                                           
                                                                                                                                            
class UNet(nn.Module):                                                                                                                               
  def __init__(self, input_channels, output_channels = 2, num_classes = 2):                                                                                  
    super().__init__()
                                                                                                                          
    self.down_channels = [
            [input_channels, 64],
            [64, 128],
            [128, 256],
            [256, 512],           
            [512, 1024]
            ]
    self.up_channels = [
            [1024, 512],
            [512, 256], 
            [256, 128], 
            [128, 64]
            ]
    self.sizes = [56, 104, 200, 392]
    self.num_classes = num_classes 

  def forward(self, x):
    print(x.shape)
    level_1_left = first_block(self.down_channels[0], x)
    print(level_1_left.shape)

    level_2_left = UNet_Down_Block(self.down_channels[1], level_1_left)
    print(level_2_left.shape)

    level_3_left = UNet_Down_Block(self.down_channels[2], level_2_left)
    print(level_3_left.shape)

    level_4_left = UNet_Down_Block(self.down_channels[3], level_3_left)
    print(level_4_left.shape)

    level_5 = UNet_Down_Block(self.down_channels[4], level_4_left)
    print(level_5.shape)

    level_4_right = UNet_Up_Block(self.up_channels[0], self.sizes[0], level_4_left, level_5)
    print(level_4_right.shape)

    level_3_right = UNet_Up_Block(self.up_channels[1], self.sizes[1], level_3_left, level_4_right)
    print(level_3_right.shape)

    level_2_right = UNet_Up_Block(self.up_channels[2], self.sizes[2], level_2_left, level_3_right)
    print(level_2_right.shape)

    level_1_right = UNet_Up_Block(self.up_channels[3], self.sizes[3], level_1_left, level_2_right)
    print(level_1_right.shape)

    result = conv1x1(level_1_right.shape[1], self.num_classes)(level_1_right)
    print(result.shape)

    return result
