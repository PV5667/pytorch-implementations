import torch
import torchvision

feature_maps = [(256, 69, 69), (256, 35, 35), (256, 18, 18), (256, 9, 9), (256, 5, 5)] # this is a configuration for fpn p3 p7 with input image size of 550, 550

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

base_anchor_generator()

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






#print(feature_map_sizes)




def corner_to_center(bboxes):
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


def boxes_to_offsets(anchors, assigned_bboxes):
  # make everything into center format
  center_anchors = corner_to_center(anchors)
  center_bboxes = corner_to_center(assigned_bboxes)

  a_cx, a_cy, a_w, a_h = center_anchors[: 0], center_anchors[: 1], center_anchors[: 2], center_anchors[: 3] 
  b_cx, b_cy, b_w, b_h = center_bboxes[: 0], center_bboxes[: 1], center_bboxes[: 2], center_bboxes[: 3]
  
  offset_x = (b_cx - a_cx)/a_w * 10
  offset_y = (b_cy - a_cy)/a_h * 10

  offset_w = torch.log(b_w/a_w) * 5 
  offset_h = torch.log(b_h/a_h) * 5 

  offsets = torch.cat((offset_x, offset_y, offset_w, offset_h), dim = 1) ##### test this out -- not sure if the dimension is 1??
  return offsets

def offsets_to_bboxes(anchors, offset_preds):
  center_anchors = corner_to_center(anchors)
  ox, oy, ow, oh = offset_preds[:, 0], offset_preds[:, 1], offset_preds[:, 2], offset_preds[:, 3]
  ax, ay, aw, ah = center_anchors[:, 0], center_anchors[:, 1], center_anchors[:, 2], center_anchors[:, 3]

  bx = ((ox * aw) / 10) + ax
  by = ((oy * ah) / 10) + ay
  bw = torch.exp(ow / 5) * aw
  bh = torch.exp(oh / 5) * ah

  bboxes = torch.cat((bx, by, bw, bh), dim = 1)
  return bboxes



def match_boxes(ground_truth, anchors, iou_thresh):
  num_bboxes, num_anchors = len(ground_truth), len(anchors)

  jaccard_matrix = torchvision.ops.box_iou(anchors, ground_truth)

  pair_map = torch.ones((num_anchors,), dtype = torch.long  ) * -1
  max_ious, indices = torch.max(jaccard_matrix, dim = 1)
  #print(indices)
  anchor = torch.nonzero(max_ious >= iou_thresh).reshape(-1)
  box = indices[max_ious >= iou_thresh]
  #print(anchor)
  #print(box)
  #print(pair_map)
  pair_map[anchor] = box
  #print(pair_map)


  fill_col = torch.ones((num_anchors, ), dtype = torch.long) * -2
  fill_row = torch.ones((num_bboxes, ), dtype = torch.long) * -2


  for i in range(num_bboxes):
    max_idx = torch.argmax(jaccard_matrix)
    #print(max_idx)
    b_j = (max_idx % num_bboxes).long()
    a_i = (max_idx / num_bboxes).long()

    pair_map[a_i] = b_j

    jaccard_matrix[:, b_j] = fill_col
    jaccard_matrix[a_i, :] = fill_row
  return pair_map
