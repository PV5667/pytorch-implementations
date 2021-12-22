def generate_boxes(data, sizes, ratios):
  # Essentially generates a set of anchor boxes for each pixel in an image
  height, width = data.shape[-2:]
  num_sizes, num_ratios = len(sizes), len(ratios)
  num_boxes_per_pixel = num_sizes + num_ratios - 1 # This is used for the num of anchors per pixel because if we went over each and every one, computational complexity would be too high
  size_tensor = torch.tensor(sizes)
  ratios_tensor = torch.tensor(ratios)

  offset_w, offset_h = 0.5, 0.5
  steps_h = 1/height
  steps_w = 1/width

  center_h = (torch.arange(height) + offset_h) * steps_h
  center_w = (torch.arange(width) + offset_w) * steps_w

  shift_y, shift_x = torch.meshgrid(center_h, center_w) # creates a grid based on the inputted center coordinates
  shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

  # width of the anchor box = ws * sqrt(r)
  # height of the anchor box = hs/sqrt(r)

  w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]), sizes[0] * torch.sqrt(ratio_tensor[1:]))) * height/width
  h = torch.cat((size_tensor/torch.sqrt(ratio_tensor[0]), sizes[0] * torch.sqrt(ratio_tensor[1:])))

  # torch.stack() --> concatenates tensors along a new dimension
  # divides anchors by 2
  anchors = torch.stack((-w, -h, w, h)).T.repeat(height * width, 1)/2
  
  # for each center point in the image, a grid of centers is generated num_boxes_per_pixel times
  out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim = 1).repeat_interleave(num_boxes_per_pixel, dim = 0)
  
  out = out_grid + anchors
  return out.unsqueeze(0)

def box_area(boxes):
  # boxes input format: (topleftX, topleftY, bottomrightX, bottomrightY)
  # The area of the box is then (bottomRightX - topLeftX)*(topLeftY - bottomRightY)
  area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 1] - boxes[:, 3])
  return area

def find_iou(boxlist1, boxlist2):
  areas_1 = box_area(boxlist1)
  areas_2 = box_area(boxlist2)

  # to get the topLefts of the intersection, we need to find the max values
  # vice versa for bottomRight points

  intersection_topLeft = torch.max(boxlist1[:, None, :2], boxlist2[:, :2])
  intersection_bottomRight = torch.min(boxlist1[:, None, 2:], boxlist2[:, 2:])

  intersections = (intersection_bottomRight - intersection_topLeft).clamp(min=0) # clamps values in range of min = 0, max = max value

  inter_area = intersections[:, :, 0] * intersections[:, :, 1] 
  union_area = areas_1[:, None] + areas_2 - inter_areas
  return inter_areas / union_areas

def anchor_bbox_match(ground_truth, anchors, iou_threshold = 0.5, device):
  num_anchors, num_bboxes = anchors.shape[0], ground_truth.shape[0]

  jaccard = find_iou(anchors, ground_truth)

  # create a dummy map of the anchor-bbox pairs -- filled with -1 for now

  pair_map = torch.full((num_anchors, ), -1, dtype = torch.long, device = device)

  max_ious, indices = torch.max(jaccard, dim = 1) # max ious and their indices

  anc_i  = torch.nonzero(max_ious >= 0.5).reshape(-1)
  box_j = indices[max_ious >= 0.5]

  pair_map[anc_i] = box_j

  dummy_col = torch.full((num_anchors, ), -1)
  dummy_row = torch.full((num_rows, ), -1)

  for _ in range(num_bboxes):
    max_idx = torch.argmax(jaccard) # index of element with max iou
    box_idx = (max_idx % num_bboxes).long()
    anc_idx = (max_idx / num_bboxes).long()

    pair_map[anc_idx] = box_idx # at index of specific anchor in pair map, there is index of ground truth box
    # fill in the row, col of the foudn anchor with dummy values
    jaccard[:, box_idx] = dummy_col
    jaccard[anc_idx, :] = dummy_row
  return pair_map



def corner_to_center(boxes):
  x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
  center_x = (x1 + x2) / 2
  center_y = (y1 + y2) / 2
  w = x2 - x1
  h = y2 - y1
  boxes = torch.stack((cx, cy, w, h), axis = -1)
  return boxes



def offset_from_bbox(anchors, bboxes, eps = 1e-6):
  c_anc = corner_to_center(anchors) # generate centers of anchors
  c_bbox = corner_to_center(bboxes) # generate center of bboxes

  # following formula on pg. 587
  offset_xy = 10 * (c_bbox[:, :2] - c_anc[:, :2])/ c_anc[:, 2:]
  offset_wh = 5 * torch.log(eps + c_bbox[:, 2:] / c_anc[:, 2:])

  offset = torch.cat([offset_xy, offset_wh], axis = 1)
  return offset


# labelling classes, offsets onto anchor boxes

def anchor_assigning(anchors, labels):
  # assigns the classes and the offsets of the ground truth boxes (obtained from anchor matching)
  # to their respective anchors
  
  batch_size, anchors = labels.shape[0], anchors.squeeze(0)
  batch_offset, batch_mask, batch_classes = [], [], []
  device, num_anchors = anchors.device, anchors.shape[0]
  for i in range(batch_size):
    label = labels[i, :, :]
    anchors_bbox_map = anchor_bbox_match(label[:, 1:], anchors, device)

    bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)

    # class labels, bounding box tensors initialized with zeroes

    classes = torch.zeros(num_anchors, dtype=torch.long, device=device)
    bbox = torch.zeros((num_anchors, 4), dtype = torch.float32, device = device)

    indices_true = torch.nonzero(anchors_bbox_map >= 0)
    bbox_index = anchors_bbox_map[indices_true]
    class_labels[indices_true] = label[bbox_index, 0].long
    bbox[indices_true] = label[bbox_index, 1:]

    # adding the offsets
    offset = offset_from_bbox(anchors, bbox) * bbox_mask
    batch_offset.append(offset.reshape(-1))
    batch_mask.append(bbox_mask.reshape(-1))
    batch_classes.append(classes)
  bbox_offset = torch.stack(batch_offset)
  bbox_mask = torch.stack(batch_mask)
  classes = torch.stack(batch_classes)
  return (bbox_offset, bbox_mask, classes)


  