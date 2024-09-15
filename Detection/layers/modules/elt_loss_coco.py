# -*- coding: utf-8 -*-
from __future__ import division
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
#from ..box_utils import IoG, decode_new
import sys



class elt_loss_coco(nn.Module):
    def __init__(self):
        super(elt_loss_coco, self).__init__()
        #self.use_gpu = use_gpu

    def forward(self,target,predict_elt,size_average=True):
        #print(predict)
        #predict_elt=np.array(predict)
        #print(predict_elt)
        #c=predict_elt.size(1)
        #nt,ht,wt=target.size()
        predict_elt = predict_elt.view(-1, 81)
        target=target.view(-1)
        loss=F.cross_entropy(predict_elt,target,size_average=size_average)
        
        return loss

