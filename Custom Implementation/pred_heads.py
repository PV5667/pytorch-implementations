"""
1. Class Predictor
2. Anchor Box Offset Predictor
3. Mask Coefficients
4. Distance Regressor
5. Keypoint Regressor (for training only)


Notes: It was observed that to apply fully connected layers for regressing the distance and the keypoints would be too computationally expensive
      due to the size of the feature maps. Thus, we have decided to use 1x1 convolutions in the interest of efficiency.

Adding in the activations as a part of the functions so I don't have to worry about them later
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

def class_pred(num_classes, num_anchors):
  # softmax will automatically be applied when cross entropy loss is used
  return nn.Conv2d(256, num_classes*num_anchors, kernel_size = (1,1), stride = (1,1), bias = False)

def box_pred(num_anchors):
  layers = nn.Sequential(nn.Conv2d(256,4*num_anchors, kernel_size = (1,1), stride = (1,1), bias = False), nn.ReLU())
  return layers

def mask_coefficients(k_masks, num_anchors):
  layers = nn.Sequential(nn.Conv2d(256, k_masks*num_anchors, kernel_size = (1,1), stride = (1,1), bias = False), nn.Tanh())
  return layers

def distance_pred(in_dimensions):
  input = in_dimensions[1] * in_dimensions[2] * in_dimensions[3]
  layers = nn.Sequential(nn.Conv2d(256, 1, kernel_size = (1,1), stride = (1, 1), bias = False), nn.Softplus())
  return layers

def xy_pred(in_dimensions):
  input = in_dimensions[1] * in_dimensions[2] * in_dimensions[3]
  layers = nn.Sequential(nn.Conv2d(256, 2, kernel_size = (1,1), stride = (1, 1), bias = False),  nn.Tanh())
  return layers

