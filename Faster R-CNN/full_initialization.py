import torch
import torchvision
import torchvision.nn.functional as F




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


def corner_to_center(bboxes):
  # fed in as TopLeftX, TopLeftY, BottomRightX, BottomRightY
  # return as centerX, centerY, width, height
  center_x = bboxes[:, :, 2] + bboxes[:, :, 0] / 2
  center_y = bboxes[:, :, 3] + bboxes[:, :, 1] / 2

  width = bboxes[:, :, 2] - bboxes[:, :, 0]
  height = bboxes[:, :, 3] - bboxes[:, :, 1]
  
  boxes_as_centers = torch.stack([center_x, center_y, width, height], dim = -1)

  return boxes_as_centers


def offsets_to_bboxes(anchors, offset_preds):
  center_anchors = corner_to_center(anchors)
  ox, oy, ow, oh = offset_preds[:, :, 0], offset_preds[:, :, 1], offset_preds[:, :, 2], offset_preds[:, :, 3]
  ax, ay, aw, ah = center_anchors[:, :, 0], center_anchors[:, :, 1], center_anchors[:, :, 2], center_anchors[:, :, 3]

  bx = ((ox * aw) / 10) + ax
  by = ((oy * ah) / 10) + ay
  bw = torch.exp(ow / 5) * aw
  bh = torch.exp(oh / 5) * ah

  bboxes = torch.cat((bx, by, bw, bh), dim = 0)
  return bboxes

def anchors(sizes, aspect_ratios):
  sizes = torch.tensor(sizes)
  aspect_ratios = torch.tensor(aspect_ratios)

  h_scales = torch.sqrt(aspect_ratios)
  w_scales = 1/h_scales

  widths = (w_scales.view(w_scales.shape[0], 1) * sizes.view(1, sizes.shape[0])).view(-1)
  heights = (h_scales.view(h_scales.shape[0], 1) * sizes.view(1, sizes.shape[0])).view(-1)

  templates = torch.stack([-widths, -heights, widths, heights], dim = 1)/2 

  return templates.round()

strides = []

feature_map_sizes = [(14, 14), (14, 14), (14, 14)]

sizes = ((128, 256, 512),)

aspect_ratios=((0.5, 1.0, 2.0),)






def stride_generator(feature_map_sizes=feature_map_sizes):
  for size in feature_map_sizes:
    h_stride = img_h // size[0] 
    w_stride = img_w // size[1]
    strides.append([torch.tensor(h_stride), torch.tensor(w_stride)])

stride_generator()

print(strides)

base_anchors = []

def base_anchor_generator(sizes=sizes, aspect_ratios=aspect_ratios):
  for size in sizes: 
    for ratio in aspect_ratios:
      base_anchors.append(anchors(size, ratio))

base_anchor_generator()



def grid_anchors(map_sizes, strides, base_anchors):
  
  output = []
  count = 0

  for i in range(len(strides)):
    stride = strides[i]
    size = map_sizes[i] 
    map_height, map_width = size
    x_stride, y_stride = stride
    

    centers_x = torch.arange(map_width) * x_stride
    centers_y = torch.arange(map_height) * y_stride

    # generate a grid of these centers

    centers_x, centers_y = torch.meshgrid(centers_x, centers_y, indexing="ij")

    # flatten resulting grids
    centers_x = centers_x.reshape(-1)
    centers_y = centers_y.reshape(-1)

    
    result = torch.stack((centers_x, centers_y, centers_x, centers_y), dim = 1)

    anchors_to_be_projected = torch.cat((base_anchors[0][i], base_anchors[0][i + 3], base_anchors[0][i + 6]))
    anchors_to_be_projected = anchors_to_be_projected.reshape(-1, 4)

    output.append((result.view(-1, 1 , 4) + anchors_to_be_projected).reshape(-1, 4))
  return output 



class Faster_RCNN(nn.Module):
  def __init__(self, in_channels, num_anchors, num_classes):
    super().__init__()
    self.num_anchors = num_anchors
    self.num_classes = num_classes
    self.VGG16 = VGG16_BackBone(in_channels)
    self.prep_conv_RPN = nn.Conv2d(512, 512, kernel_size = (3,3), padding = 1)

    self.fc = nn.Sequential(nn.Conv2d(512, 4096, kernel_size=(1,1), stride = (1,1), bias= False), nn.ReLU(), nn.Conv2d(4096, 4096, kernel_size=(1,1), stride = (1,1), bias= False), nn.ReLU())

    self.final_class_layer = nn.Conv2d(4096, self.num_classes, kernel_size = (1, 1), stride = (1,1), bias = False)
    self.final_bbox_layer = nn.Conv2d(4096, 4, kernel_size = (1,1), stride = (1,1), bias = False)
    # define the prediction heads, use result of nms, take its .size()[0]

  def forward(self, x):
    feature_map = self.VGG16(x)
    # Run this feature map through the RPN and a conv layer
    print("Feature map shape: ", feature_map.shape)

    # RPN
    RPN_map = F.relu(self.prep_conv_RPN(feature_map))
    print("RPN input map shape: ", RPN_map.size())
    # generate template anchors
    template_anchors = grid_anchors(feature_map_sizes, strides, base_anchors)
    #anchors = {GENERATE TEMPLATE ANCHORS HERE FOR THE FEATURE MAP}
    binary_preds = binary_pred(self.num_anchors)(RPN_map)
    

    binary_preds = binary_preds.reshape(binary_preds.shape[0], -1, 1)
    binary_preds = binary_preds.view(-1)

    print(binary_preds.shape)

    offset_preds = box_pred(self.num_anchors)(RPN_map)
    offset_preds = offset_preds.reshape(offset_preds.shape[0], -1, 4)

    #print(len(template_anchors)) 
    #print(offset_preds.shape)

    template_anchors = torch.cat(template_anchors, dim = 0)
    template_anchors = template_anchors.view(1, -1, 4)

    print(template_anchors.shape)
    # use nms built-in function to filter predictions
    boxes_converted = offsets_to_bboxes(template_anchors, offset_preds)
    boxes_converted = boxes_converted.view(-1, 4)

    print("Boxes", boxes_converted.shape)
    
    filtered_preds = torchvision.ops.nms(boxes_converted, binary_preds, 0.7)

    filtered_preds = filtered_preds.view(-1, 5).float()
    print(filtered_preds.shape)
    # use roi pooling built-in function  


    result = torchvision.ops.roi_pool(feature_map, filtered_preds, (7, 7))

    print(result.shape)

    result = self.fc(result)
    print(result.shape)

    class_preds = self.final_class_layer(result)
    bbox_preds = self.final_bbox_layer(result)

    return class_preds, bbox_preds 



net = Faster_RCNN(3, 9, 10)
t = torch.randn(1,3, 224, 224)
net.forward(t)
