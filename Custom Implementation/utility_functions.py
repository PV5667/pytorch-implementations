"""
Concatenating predictions, Fast NMS, anchor generation -- oh boy!!!

"""


import torchvision
import torch.nn.functional as F
import math






def fast_nms(bboxes, class_preds, mask_coefficients, num_classes, distance_preds, xy_preds, iou_thresh = 0.02, training = True):
  #print(class_preds)

  #print(class_preds[:10])
  preds, classes = class_preds.max(dim = 1)
  #print(classes[:20])
  print(preds)
  preds_sorted, indices = preds.sort(dim = 0, descending = True)
  preds_sorted_confidences = F.softmax(preds_sorted)
  preds_sorted_confidences = preds_sorted_confidences[:200]
  #print(indices)
  indices = indices[:200]
  #print(indices)
  boxes = bboxes[indices]
  #print(boxes.shape)
  jaccard_matrix = torchvision.ops.box_iou(boxes, boxes)
  print(jaccard_matrix)

  """Using triu to eliminate any redundant ious; _ means that it is inplace operation"""
  jaccard_matrix.triu_(diagonal = 1)
  max_values, inds = jaccard_matrix.max(dim = 0)
  print(max_values)

  indices = indices[max_values <= iou_thresh ]

  print(indices)
  print(indices.shape)

  boxes_out = bboxes[indices]
  classes_out = classes[indices]
  masks_coeffs_out = mask_coefficients[indices]
  class_preds_out = class_preds[indices]
  distances_out = distance_preds[torch.unique(torch.floor(indices.div(3)).long())]
  print(distances_out.shape)
  if training == True:
    xy_out = xy_preds[torch.unique(torch.floor(indices.div(3)).long())]
    print(xy_out.shape)
    return boxes_out, class_preds_out, masks_coeffs_out, distances_out, xy_out

  return boxes_out, class_preds_out, masks_coeffs_out, distances_out
  




net = CompleteNetwork(3, 8, 10, 3)
t = torch.randn(2, 3, 550, 550)
bbox_preds, class_preds, coefficients, prototype_masks, distances, xy_preds = net.forward(t)

bbox_preds = torch.randn(2, 19248, 4)
batch_size = 2



for i in range(batch_size):
  fast_nms(bbox_preds[i], class_preds[i], coefficients[i], 10, distances[i], xy_preds[i])



def concat(preds):
  results = []
  for p in preds:
    results.append(torch.flatten(p.permute(0, 2, 3, 1), start_dim = 1))
  return torch.cat(results, dim = 1)


