import torch
import torchvision                                                                                                                                      
import torch.nn as nn                                                                                                                       
                                                                                                                                            
                                                                                                                                            
                                                                                                                                            
class UNET_Block():                                                                                                                         
  def __init__(self, input_channels, output_channels):                                                                                      
    """Initializes one of the blocks of the U-Net architecture."""                                                                          
    super().__init__()                                                                                                                      
    self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3,3), stride = (1,1), padding = (1, 1), bias = False)                                                                        
    self.conv2 = nn.Conv2d(input_channels, output_channels, kernel_size=(3,3), stride = (1,1), padding = (1, 1), bias = False)                                                                        
                                                                                                                                            
                                                                                                                                            
class UNet():                                                                                                                               
  def __init__(self, input_channels, output_channels = 2):                                                                                  
    super().__init__()                                                                                                                      
    self.maxpool = nn.MaxPool2d(kernel_size = (3, 3), stride = 2)     

