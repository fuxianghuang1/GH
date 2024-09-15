import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils_wang import decode,decode_pred_yiou

class angcos(nn.Module):
    def __init__(self):
        super(angcos, self).__init__()
        self.variance = [0.1,0.2]
    def forward(self, pred, target, priors_iou,beta=5):
        decoded_boxes=decode(pred,priors_iou,self.variance)

        #print(decoded_boxes.requires_grad, target.requires_grad)
        #print(decoded_boxes.shape, target.shape)
        #print(decoded_boxes, target)
        input_vevtor=decoded_boxes[:,2:]-decoded_boxes[:,0:2]
        target_vevtor=target[:,2:]-target[:,0:2]
        #print(input_vevtor,input_vevtor.shape)
        #print(target_vevtor,target_vevtor.shape)
        similarity = F.cosine_similarity(input_vevtor, target_vevtor, dim=1)
        #print(similarity.shape)
        #print(similarity)
        cos_loss = 1 - similarity
        #print(kkkkkk)
        return cos_loss*beta

