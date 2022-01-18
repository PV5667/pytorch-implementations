def binary_pred(num_anchors):
  # softmax will automatically be applied when cross entropy loss is used
  return nn.Sequential(nn.Conv2d(512, num_anchors, kernel_size = (1,1), stride = (1,1), bias = False), nn.Sigmoid())

def box_pred(num_anchors):
  layers = nn.Sequential(nn.Conv2d(512,4*num_anchors, kernel_size = (1,1), stride = (1,1), bias = False), nn.ReLU())
  return layers