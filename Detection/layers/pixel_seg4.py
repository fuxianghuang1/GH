import torch
import numpy as np
import torch.nn as nn
import time
from torch.autograd import Variable
def score_label(input_image,targets,use_gpus,th1,th2):
    #t0=time.time()
    n,c,w,h=input_image.size()
    score_label=torch.zeros(n,w,h)
    for idx in range(n):
        areas,coor,labels=size(targets[idx],w,h)
        #print(labels)
        #print(areas)
        labels=labels
        bareas=torch.sort(-areas)#sort gain sort and original index
        #print(bareas)
        #print(bareas[0])
        #print(bareas[1])
        areas1=((-bareas[0]<th2) * (-bareas[0]>=th1)).nonzero()#select bare
        #print((-bareas[0]<th2) * (-bareas[0]>th1))
        #print(bareas)
        #print(areas1)
        #print(areas2)
        #print(areas1.shape,'...',areas1.dim)
        #print(coor)
        if areas1.shape==torch.Size([]):
            score_label[idx]=score_label[idx]
        else:
            for i in areas1:
                #print(bareas[1][i].shape)#the index before sort
                #print(labels[bareas[1][i][0]]) #label
                #print(coor[1][bareas[1][i][0]])
                '''print(score_label.shape)
                print(coor[0].shape,coor[1].shape,coor[2].shape,coor[3].shape)
                print(bareas[1].shape)
                print(labels.shape)'''
                #print(int(coor[0][bareas[1][i][0]]))
                score_label[idx,int(coor[0][bareas[1][i][0]]):int(coor[1][bareas[1][i][0]]+1),int(coor[2][bareas[1][i][0]]):int(coor[3][bareas[1][i][0]]+1)]=labels[bareas[1][i][0]]
    #t1=time.time()
    #print('told=%.4f'%(t1-t0))
    #print(dddd)
    return score_label

def size(targets,width,height): 
    xmin=targets[:,0]*width
    xmax=targets[:,2]*width
    ymin=targets[:,1]*height
    ymax=targets[:,3]*height
    labels=targets[:,4]
    w=xmax-xmin
    h=ymax-ymin
    #print(w,h)
    area=w*h
    return area.data ,(xmin.data,xmax.data,ymin.data,ymax.data),labels.data

def downsample(input_image,targets,use_gpus,th1,th2,width,height,steps):
    #print(input_image.type)
    score_label1=score_label(input_image,targets,use_gpus,th1,th2)
    n,w,h=score_label1.shape
    output=torch.zeros(n,width,height)
    #print(output.shape)
    #score_label1=score_label1.unsqueeze(1)
    #score_label1=Variable(score_label1.cuda())
    #print(score_label1.size())
    #maxpool=nn.MaxPool2d(steps,stride=8)
    #for idm in range(n):
    #output=maxpool(score_label1) 
    #output=output.long()
    #print (output)
    output[0:n, 0:width, 0:height] = score_label1[0:n,int(0.5*steps):int((width-0.5)*steps)+1:steps,int(0.5*steps):int((height-0.5)*steps)+1:steps]
    #output=Variable(output.cuda())
    #print(output.type())
    output=Variable(output.cuda())
    output=output.long()  
    #print(output.shape) 
    return output

    
        

 
