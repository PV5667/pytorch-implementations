class ProtoNet(nn.Module):
  """Specify the number of masks to be outputted by Protonet with k as a hyperparameter
     Different results are shown in terms of FPS and AP in Table 2b: https://arxiv.org/pdf/1912.06218.pdf
  """
  def __init__ (self, k, input_channels=256):
    super().__init__()
    self.conv = nn.Conv2d(input_channels, input_channels, kernel_size = (3,3), stride = (1,1), padding = 1)
    self.upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
    self.conv1x1 = nn.Conv2d(input_channels, k, kernel_size = (1,1), stride = (1,1), bias = False) 
  def forward(self, x):
    x = F.relu(self.conv(x))
    x = F.relu(self.conv(x))
    x = F.relu(self.conv(x))
    print(x.shape)
    x = F.relu(self.upsample(x))
    print(x.shape)
    x = F.relu(self.conv1x1(x))
    print(x.shape)
    return x


net = ProtoNet(8)

t = torch.randn(1, 256, 69, 69)

net.forward(t)
