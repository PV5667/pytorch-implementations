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


############### anchor box functions ##################




import torch


feature_maps = [(256, 69, 69), (256, 35, 35), (256, 18, 18), (256, 9, 9), (256, 5, 5)]

data_img_size = (3, 550, 550)
img_w, img_h = data_img_size[-2:]



def anchors(sizes, aspect_ratios):
  sizes = torch.tensor(sizes)
  aspect_ratios = torch.tensor(aspect_ratios)

  h_scales = torch.sqrt(aspect_ratios)
  w_scales = 1/h_scales

  widths = (w_scales.view(w_scales.shape[0], 1) * sizes.view(1, sizes.shape[0])).view(-1)
  heights = (h_scales.view(h_scales.shape[0], 1) * sizes.view(1, sizes.shape[0])).view(-1)

  templates = torch.stack([-widths, -heights, widths, heights], dim = 1)/2 

  return templates.round()
sizes = ((24, 48, 96, 192, 384),)

aspect_ratios=((0.5, 1.0, 2.0),)



base_anchors = []



def base_anchor_generator(sizes=sizes, aspect_ratios=aspect_ratios):
  for size in sizes: 
    for ratio in aspect_ratios:
      base_anchors.append(anchors(size, ratio))

base_anchor__generator()

print(base_anchors[0])



strides = []

feature_map_sizes = [(69, 69), (35, 35), (18, 18), (9, 9), (5, 5)]

def stride_generator(feature_map_sizes=feature_map_sizes):
  for size in feature_map_sizes:
    h_stride = img_h // size[0] 
    w_stride = img_w // size[1]
    strides.append([torch.tensor(h_stride), torch.tensor(w_stride)])

stride_generator()


print(strides)



count = 0


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
    #print(centers_x.shape)
    #print(centers_y.shape)

    # flatten resulting grids
    centers_x = centers_x.reshape(-1)
    centers_y = centers_y.reshape(-1)

    
    result = torch.stack((centers_x, centers_y, centers_x, centers_y), dim = 1)

    #print(result.view(-1, 1, 4))
    anchors_to_be_projected = torch.cat((base_anchors[0][i], base_anchors[0][i + 5], base_anchors[0][i + 10]))
    anchors_to_be_projected = anchors_to_be_projected.reshape(-1, 4)


    output.append((result.view(-1, 1 , 4) + anchors_to_be_projected).reshape(-1, 4))
  return output 




def bbox_to_center(bboxes):
  # fed in as TopLeftX, TopLeftY, BottomRightX, BottomRightY
  # return as centerX, centerY, width, height
  center_x = bboxes[:, 2] + bboxes[:, 0] / 2
  center_y = bboxes[:, 3] + bboxes[:, 1] / 2

  width = bboxes[:, 2] - bboxes[:, 0]
  height = bboxes[:, 3] - bboxes[:, 1]
  
  boxes_as_centers = torch.stack([center_x, center_y, width, height], dim = -1)

  return boxes_as_centers



def center_to_corner(bboxes):
  # fed in as centerX, centerY, width, height
  # return as TopLeftX, TopLeftY, BottomRightX, BottomRightY
  center_x, center_y, width, height = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

  top_left_x = center_x - width * 0.5
  top_left_y = center_y - height * 0.5
  bottom_right_x = center_x + width * 0.5
  bottom_right_y = center_y + height * 0.5

  boxes_as_corners = torch.stack([top_left_x, top_left_y, bottom_right_x, bottom_right_y], dim = -1)

  return boxes_as_corners

test = grid_anchors(feature_map_sizes, strides, base_anchors)

for i in test:
  print("-------------||||||||||---------")
  print(bbox_to_center(i).shape)
