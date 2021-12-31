"""


Train the network
Known hyperparameters: 
k(# of protoype masks to generate) --> default value: 8 masks


"""




#Loss calculating functions


""" 
losses needed for the network: 

Bounding boxes: smooth L1 loss
Classes: CrossEntropy Loss
Masks : Pixel-wise Binary CrossEntropy 
Distance: smooth L1 loss
Keypoint: custom projection loss --> implement this

the combined loss of this network will be L(bbox) + L(classes) + L(Masks) + L(distance) + L(keypoint)

""" 

import torch
import torch.nn as nn



bbox_alpha = 1
conf_alpha = 1.5
mask_alpha = 6.125

# TODO: implement the "match" function here at https://github.com/dbolya/yolact/blob/57b8f2d95e62e2e649b382f516ab41f949b57239/layers/box_utils.py#L159




def bbox_L1_loss(pos_idx, bbox_preds, bbox_targets):
  bbox_preds = bbox_preds[pos_idx].reshape(-1, 4)
  bbox_targets = loc_t[pos_idx].reshape(-1, 4)
  bbox_loss = nn.SmoothL1Loss(bbox_preds, loc_t, reduction = "sum") * bbox_alpha
  return bbox_loss



def combined_class_box_loss(predictions):
  """Predictions will be inputted as bbox offsets, class labels, mask tensors, 3D points"""



def keypoint_loss(projection_matrix, keypoint_pred, distance_pred, gt_keypoint, gt_distance):
  combined_3d_point = torch.tensor(keypoint_pred[0], keypoint_pred[1], distance_pred)
  x = torch.mm(projection_matrix, combined_3d_point)
  out = (1/distance_pred) * torch.linalg.norm(x, dim=1, ord=2)
  loss = out.mean()
  return loss

def total_loss(projection_matrix, keypoint_pred, distance_pred, gt_keypoint, gt_distance ):
  L_keypoint = keypoint_loss(projection_matrix, keypoint_pred, distance_pred, gt_keypoint, gt_distance)
  L_bbox = nn.SmoothL1Loss()
  L_distance = nn.SmoothL1Loss(distance_pred, gt_distance)
  L_classes = nn.CrossEntropyLoss()
  L_masks = nn.BCEWithLogitsLoss()

  total_loss = L_keypoint + L_bbox + L_distance + L_classes + L_masks
  return total_loss



