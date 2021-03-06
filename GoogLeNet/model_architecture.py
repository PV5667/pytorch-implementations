from torch.nn.modules.activation import ReLU
import torchvision
import torch 
import torch.nn as nn
import torch.nn.functional as F

############ output_channels = [1x1, 3x3 reduce, 3x3, 5x5 reduce, 5x5, pool proj]

class Inception_Module(nn.Module):
    def __init__ (self, input_channels, output_channels):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(input_channels, output_channels[0], kernel_size = 1, stride = (1,1), bias = False), nn.ReLU())
        self.branch2 = nn.Sequential(nn.Conv2d(input_channels, output_channels[1], kernel_size = 1, stride = (1,1), bias = False), 
                                        nn.ReLU(), nn.Conv2d(output_channels[1], output_channels[2], kernel_size = 3, stride = (1,1), padding = (1,1), bias = False), 
                                        nn.ReLU())
        self.branch3 = nn.Sequential(nn.Conv2d(input_channels, output_channels[3], kernel_size = 1, stride = (1,1), bias = False), 
                                        nn.ReLU(), nn.Conv2d(output_channels[3], output_channels[4], kernel_size = 5, stride = (1,1), padding = (2,2), bias = False), 
                                        nn.ReLU())
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size = 3, stride = (1,1)), nn.Conv2d(input_channels, output_channels[5], kernel_size = 1, stride = (1,1), padding = (1,1), bias = False), nn.ReLU())

    def forward(self, x):
        a = self.branch1(x)
        b = self.branch2(x)
        c = self.branch3(x)
        d = self.branch4(x)
        x = torch.cat([a, b, c, d], dim = 1)
        return x 


class GoogLeNet(nn.Module):
    def __init__ (self, input_channels, output_channels):
        super().__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size = (7,7), stride = (2,2), padding = (3,3), bias = False), 
                                         nn.ReLU(), nn.MaxPool2d(kernel_size = 3, stride = (2, 2), padding = (1,1)), nn.BatchNorm2d(64), 
                                         nn.Conv2d(64, 64, kernel_size=(1,1), stride = (1,1),  bias = False), nn.ReLU(),
                                         nn.Conv2d(64, 192, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False), nn.ReLU(), 
                                         nn.BatchNorm2d(192), nn.MaxPool2d(kernel_size = 3, stride = (2,2), padding = (1,1)))
        self.inception3a = Inception_Module(192, [64, 96, 128, 16, 32, 32])
        self.inception3b = Inception_Module(256, [128, 128, 192, 32, 96, 64])
        self.maxpool1 = nn.MaxPool2d(kernel_size =3, stride = 2, padding = (1,1))
        self.inception4a = Inception_Module(480, [192, 96, 208, 16, 48, 64])
        self.inception4b = Inception_Module(512, [160, 112, 224, 24, 64, 64])
        self.inception4c = Inception_Module(512, [128, 128, 256, 24, 64, 64])
        self.inception4d = Inception_Module(512, [112, 144, 288, 32, 64, 64])
        self.inception4e = Inception_Module(528, [256, 160, 320, 32, 128, 128])
        self.maxpool2 = nn.MaxPool2d(kernel_size =3, stride = 2, padding = (1,1))
        self.inception5a = Inception_Module(832, [256, 160, 320, 32, 128, 128])
        self.inception5b = Inception_Module(832, [384, 192, 384, 48, 128, 128])
        self.avgpool1 = nn.AvgPool2d(kernel_size = 7, stride = (1,1))
        self.dropout1 = nn.Dropout2d(0.4)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 1000)

        self.pred_branch1 = nn.Sequential(nn.AvgPool2d(kernel_size = 5, stride = 3), 
                                         nn.Conv2d(512, 128, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False), 
                                         nn.ReLU(), nn.Flatten(), nn.Linear(2048, 1024), nn.Dropout2d(0.7), nn.Linear(1024, 1000))
        self.pred_branch2 = nn.Sequential(nn.AvgPool2d(kernel_size = 5, stride = 3), 
                                         nn.Conv2d(528, 128, kernel_size = 3, stride = (1,1), padding = (1,1), bias = False), 
                                         nn.ReLU(), nn.Flatten(), nn.Linear(2048, 1024), nn.Dropout(0.7), nn.Linear(1024, 1000))


        self.training = True
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.inception3a(x)

        x = self.inception3b(x)
    
        x = self.maxpool1(x)

        x = self.inception4a(x)

        # add pred branch 1 here
        if self.training == True:
            pred1 = self.pred_branch1(x)

        x = self.inception4b(x)
        
        x = self.inception4c(x)
        
        x = self.inception4d(x)
  
        # add pred branch 2
        if self.training == True:
            pred2 = self.pred_branch2(x)
        
        x = self.inception4e(x)

        x = self.maxpool2(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool1(x)
        
        x = self.dropout1(x)
        
        x = self.flatten(x)
        
        x = self.linear1(x)
        print(x.size())
        if self.training == True:
            return pred1, pred2, x
        else:
            return x

        

net = GoogLeNet(3, 5)
t = torch.randn(1, 3, 224, 224)
print(t.size())
net.forward(t)



