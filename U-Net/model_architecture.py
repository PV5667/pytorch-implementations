import torch                                                                                                                                                                                                                                                           
import torchvision                                                                                                                                      
import torch.nn as nn  
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop


def first_block(channels, input):
  input_channels = channels[0]
  output_channels = channels[1]
  layers = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size = (3,3), stride = (1,1),  bias = False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace = True),
                         nn.Conv2d(output_channels, output_channels, kernel_size = (3,3), stride = (1,1), bias = False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace = True))
  return layers(input)

def reverse_conv(output_size):
  """Simply calculates the input size for each unet block when given the output size"""
  input_size = output_size + 4
  return input_size


def get_input_size(output_size):
  """Given the expected output size of the network, the input size is outputted"""
  for _ in range(4):
    output_size = reverse_conv(output_size) / 2
  for _ in range(4):
    output_size = reverse_conv(output_size) * 2
  input_size = output_size + 4
  return input_size

def calculate_padding(input_size, output_size):
  padding = (input_size - output_size) / 2


def UNet_Down_Block(channels, from_up):
  """Define the UNet repeating blocks(downward convolutions)"""

  maxpool_layer = nn.MaxPool2d(kernel_size = (2, 2), stride = 2)
  result = maxpool_layer(from_up)

  input_channels = channels[0]
  output_channels = channels[1]
  layers = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size = (3,3), stride = (1,1),  bias = False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace = True),
                         nn.Conv2d(output_channels, output_channels, kernel_size = (3,3), stride = (1,1), bias = False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace = True))
  return layers(result)                                                                                                                                      


def UNet_Up_Block(channels from_left, from_bottom):
  """
  Define the UNet repeating blocks(upward convolutions)
  from_left: the tensor from the corresponding level of the downward path
  from_bottom: the tensor from the immediately previous layer of the network
  
  from_left will be center cropped and from_bottom will be upsampled. 
  Both will then be concatenated.
  """

  from_left = center_crop(from_left, [from_bottom.shape[2] * 2, from_bottom.shape[2] * 2 ])
  from_bottom = conv_transpose(from_bottom.shape[1], from_left.shape[1])(from_bottom)

  result = torch.concat((from_bottom, from_left), 1) 

  input_channels = channels[0]
  output_channels = channels[1]

  layers = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size = (3,3), stride = (1,1),  bias = False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace = True),
                         nn.Conv2d(output_channels, output_channels, kernel_size = (3,3), stride = (1,1), bias = False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace = True))
  return layers(result)         

                                                                                                                                            
def conv1x1(input_channels, output_channels):
  return nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size = (1,1), stride = (1, 1), bias = False))


def conv_transpose(input_channels, output_channels):
  return nn.ConvTranspose2d(input_channels, output_channels, kernel_size = 2, stride = 2)

                                                                                                                                  
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

    level_2_left = UNet_Down_Block(self.down_channels[1], level_1_left)

    level_3_left = UNet_Down_Block(self.down_channels[2], level_2_left)

    level_4_left = UNet_Down_Block(self.down_channels[3], level_3_left)

    level_5 = UNet_Down_Block(self.down_channels[4], level_4_left)

    level_4_right = UNet_Up_Block(self.up_channels[0], self.sizes[0], level_4_left, level_5)

    level_3_right = UNet_Up_Block(self.up_channels[1], self.sizes[1], level_3_left, level_4_right)

    level_2_right = UNet_Up_Block(self.up_channels[2], self.sizes[2], level_2_left, level_3_right)

    level_1_right = UNet_Up_Block(self.up_channels[3], self.sizes[3], level_1_left, level_2_right)

    result = conv1x1(level_1_right.shape[1], self.num_classes)(level_1_right)
    
    return result
