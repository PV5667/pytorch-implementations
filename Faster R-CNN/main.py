# The main file where the main network is defined and trained, using imported classes



"""
for each feature map outputted by the FPN, a convolution of 3 x 3 with a padding of 1 is applied
this is to make sure that jthe result has "c" channels, which is then fed into the anchor prediction (c is the RGB, etc, etc)

from the result of this convolution, anchor boxes are generated centered on each pixel
One difference to notice here because we are using FPN is that there is no need for multiscale anchor boxes, as this is taken by FPN being inherently multiscale
Furthermore, in this implementation, we will not be adding the P6 layer as a feature map to input to the RPN

The anchor boxes are then fed into two conv2D layers; one is for predicting thej objectness score of the anchor box (this is binary) and the other is for predicting the offsets to the anchors
These predictions are then fed into non-max suppression, after which they are fed into RoI pooling(check Fast RCNN) 
The formula for the ROI layer's height and width for layer P(k) is: k = k(0) + log2(sqrt(wh)/224). 224 is the scale of the training image (if using ImageNet). k(0) = 4
After this, they are sent through fully connected layers which will output class predictions 




Useful functions from the pytorch libraries in this case:

1. torchvision.ops.batched_nms
2. torchvision.ops.box_area
3. torchvision.ops.box_convert
4. torchvision.ops.box_iou
5. torchvision.ops.generalized_box_iou
6. torchvision.ops.nms
7. torchvision.ops.roi_pool



implement an RPN and an Faster RCNN class --> for 4-step alternating training

"""




class Faster_RCNN(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.FPN50 = ResNet50FPN(in_channels)
    self.prep_conv = nn.Conv2d(256, 256, kernel_size = (3,3), padding = 1)

  def forward(self, x):
    m5, m4, m3, m2 = self.FPN50(x)
    feature_maps = [m5, m4, m3, m2]
    for i in range(len(feature_maps)):
      anchors = generate_boxes(feature_maps[i], sizes = [1], ratios = [0.5, 1, 2]) ######### figure out how to implement the specific anchor size ASAP
      


      

