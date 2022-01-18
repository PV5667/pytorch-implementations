def binary_pred(num_anchors):
  # softmax will automatically be applied when cross entropy loss is used
  return nn.Sequential(nn.Conv2d(512, num_anchors, kernel_size = (1,1), stride = (1,1), bias = False), nn.Sigmoid())

def box_pred(num_anchors):
  layers = nn.Sequential(nn.Conv2d(512,4*num_anchors, kernel_size = (1,1), stride = (1,1), bias = False), nn.ReLU())
  return layers


  """
for each feature map outputted by the FPN, a convolution of 3 x 3 with a padding of 1 is applied
this is to make sure that the result has "c" channels, which is then fed into the anchor prediction (c is the RGB, etc, etc)

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


"""