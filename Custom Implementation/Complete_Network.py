# anchor ratios: (1,1), (1, 2), (2, 1)
# anchor scales: (1) --> only happens at one scale

import torch
import torch.nn as nn
import torch.nn.functional as F

class CompleteNetwork():
  def __init__(self, in_channels, k_masks, num_classes, num_anchors):
    self.FPN = ResNet50FPN_P37(in_channels)
    self.protonet = ProtoNet(k_masks)
    self.training = True
    self.num_classes = num_classes
    self.num_anchors = num_anchors
    self.k_masks = k_masks
    self.prep_conv = nn.Conv2d(256, 256, kernel_size = (3,3), stride = (1,1), padding = 1)

  def forward(self, x):
    num_classes = self.num_classes
    num_anchors = self.num_anchors
    k_masks = self.k_masks
    p7, p6, p5, p4, p3 = self.FPN(x)
    prototype_masks = self.protonet(p3)
    feature_maps = [p7, p6, p5, p4, p3]
    count = 0
    class_preds = []
    bbox_preds = []
    coefficients = []
    distances = []
    xy_preds = []

    for map in feature_maps:
      count += 1
      map = self.prep_conv(map)
      input_dimensions = tuple(map.shape)
      flattened_dim = input_dimensions[1] * input_dimensions[2] * input_dimensions[3]
      #print(flattened_dim)

      class_preds.append(class_pred(num_classes, num_anchors)(map))

      bbox_preds.append(box_pred(num_anchors)(map))

      coefficients.append(mask_coefficients(k_masks, num_anchors)(map))
      
      distances.append(distance_pred(input_dimensions)(map))

      if self.training == True:
        xy_preds.append(xy_pred(input_dimensions)(map))
      #print(count, tuple(map.shape)[-3:])
    #print(concat(bbox_preds).shape)
    #print(concat(class_preds).shape)
    #print(concat(distances).shape)

    #for pred in bbox_preds:
    #  print("BBox: ", pred.shape)
    bbox_preds = concat(bbox_preds)
    bbox_preds = bbox_preds.reshape(bbox_preds.shape[0], -1, 4)
    print(bbox_preds.shape)

    class_preds = concat(class_preds)
    class_preds = class_preds.reshape(class_preds.shape[0], -1, num_classes)
    print(class_preds.shape)

    distances = concat(distances)
    distances = distances.reshape(distances.shape[0], -1, 1)
    print(distances.shape)

    coefficients = concat(coefficients)
    coefficients = coefficients.reshape(coefficients.shape[0], -1, k_masks)
    print(coefficients.shape)

    if self.training == True:
      #print(concat(xy_preds).shape)
      xy_preds = concat(xy_preds)
      xy_preds = xy_preds.reshape(xy_preds.shape[0], -1, 2)
      print(xy_preds.shape)
      
      #return bbox_preds, class_preds, coefficients, prototype_masks, distances, xy_preds
    
    #return bbox_preds, class_preds, coefficients, prototype_masks, distances
# TODO: implement fast nms, anchor box functions, concatenating predictions


net = CompleteNetwork(3, 8, 10, 3)
t = torch.randn(2, 3, 550, 550)
net.forward(t)


