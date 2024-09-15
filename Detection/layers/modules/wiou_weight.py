import torch
from torch import nn
import torch.nn.functional as F
from utils.box_utils_wang import decode,decode_pred_yiou
eps=0.00000001
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter+eps
    ious=inter / union
    ious = torch.clamp(ious,min=0,max = 1.0)
    return ious  # [A,B]

def bbox_overlaps_iou(bboxes1, bboxes2):

    rows = bboxes1.shape[0]

    cols = bboxes2.shape[0]

    ious = torch.zeros((rows, cols))

    if rows * cols == 0:

        return ious

    exchange = False

    if bboxes1.shape[0] > bboxes2.shape[0]:

        bboxes1, bboxes2 = bboxes2, bboxes1

        ious = torch.zeros((cols, rows))

        exchange = True

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (

        bboxes1[:, 3] - bboxes1[:, 1])

    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (

        bboxes2[:, 3] - bboxes2[:, 1])



    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])

    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])



    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)

    inter_area = inter[:, 0] * inter[:, 1]+eps

    union = area1+area2-inter_area+eps

    ious = inter_area / union

    ious = torch.clamp(ious,min=0,max = 1.0)

    return ious

class wiou_weight(nn.Module):
    def __init__(self):
        super(wiou_weight, self).__init__()
        self.variance = [0.1,0.2]
    def forward(self, pred, target,alph, priors_iou,beta=5):
        decoded_boxes=decode(pred,priors_iou,self.variance)
        ious=bbox_overlaps_iou(decoded_boxes,target)
        input_vector=decoded_boxes[:,2:]-decoded_boxes[:,0:2]
        target_vector=target[:,2:]-target[:,0:2]
        similarity = F.cosine_similarity(input_vector, target_vector, dim=1)
        cos_loss = 1 - similarity
        target_xy=(target[:,:2]+target[:,2:])/2.0
        priors_xy=priors_iou[:,:2]
        weighted=1.0+torch.exp(-10*((target_xy[:,0]-priors_xy[:,0])**2+(target_xy[:,1]-priors_xy[:,1])**2))
        wiou_loss=(1.0-ious**2)*alph

        return wiou_loss, weighted, cos_loss*beta
