"""
Concatenating predictions, Fast NMS, anchor generation -- oh boy!!!

"""

def jaccard_matrix(boxes_1, boxes_2):
  # boxes input format: (topleftX, topleftY, bottomrightX, bottomrightY)
  # The area of the box is then (bottomRightX - topLeftX)*(bottomRightY - topleftY)
  area_1 = (boxes_1[:, 2] - boxes_1[:, 0])* (boxes_1[:, 3] - boxes_1[:, 1])
  area_2 = (boxes_2[:, 2] - boxes_2[:, 0])* (boxes_2[:, 3] - boxes_2[:, 1])

  




def concat(preds):
  results = []
  for p in preds:
    results.append(torch.flatten(p.permute(0, 2, 3, 1), start_dim = 1))
  return torch.cat(results, dim = 1)


