import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers import *
from .base_models import vgg, vgg_base
from data import voc_sgws, coco320


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        #self.up_size = up_size

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        #if self.up_size > 0:
            #x=F.interpolate(input=x,size=(self.up_size, self.up_size), mode='bilinear')
        return x

class BasicdeConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, relu=False,
                 bn=False, bias=True):
        super(BasicdeConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        #self.up_size = up_size

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        #if self.up_size > 0:
            #x=F.interpolate(input=x,size=(self.up_size, self.up_size), mode='bilinear')
        return x

class SSD(nn.Module):
    def __init__(self, phase,extras,num_classes, size):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.phase = phase
        # TODO: implement __call__ in PriorBox
        self.cfg = (coco320, voc_sgws)[num_classes == 21]
        #self.priorbox = PriorBox(self.cfg)
        #self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.base = nn.ModuleList(vgg(vgg_base['320'], 3))
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.loc=nn.ModuleList([nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(1024, 32, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(256, 24, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1), \
                                ])
        self.conf=nn.ModuleList([nn.Conv2d(512, num_classes*8, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(1024, num_classes*8, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(512, num_classes*8, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(256, num_classes*6, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(256, num_classes*4, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(256, num_classes*4, kernel_size=3, stride=1, padding=1), \
                                ])

        
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            #self.detect = Detect(num_classes, 0, 400, 0.01, 0.45)
    """def deconv_add(self,backward,forward,out_channels1=256,kernel_size1=3,stride1=1,padding1=0):
        print(backward.out_channels)
        deconv1=BasicdeConv(backward.out_channels, out_planes=out_channels1, kernel_size=kernel_size1, stride=stride1, padding=padding1)
        conv1=BasicConv(deconv1.out_channels,out_planes=out_channels1,kernel_size=3,stride=1,padding=1,relu=False,bn=True)
        return (forward+conv1)"""
    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources1 = list()#origin ssd sources
        sources2 = list()#deconv layers
        sources=list()#predction layers
        loc = list()
        conf = list()
        mask=list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)

        s = self.L2Norm(x)
        sources1.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)
        sources1.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x))
            if k % 2 == 1:
                sources1.append(x)
                
      
        '''p2=BasicConv
        p2=p1+self.deconv_add(p1,sources1[-2],256,3,1,0)
        sources.append(p1)
        p3=self.deconv_add(p2,sources1[-3],out_channels1=256,kernel_size1=3,stride1=1,padding1=0)
        sources.append(p1)
        p4=self.deconv_add(p3,sources1[-4],out_channels1=256,kernel_size1=2,stride1=2,padding1=0)
        sources.append(p1)
        p5=self.deconv_add(p4,sources1[-5],out_channels1=512,kernel_size1=2,stride1=2,padding1=0)
        sources.append(p1)
        p6=self.deconv_add(p4,sources1[-5],out_channels1=512,kernel_size1=2,stride1=2,padding1=0)
        sources.append(p1)'''
        # apply multibox head to source layers
        for (x, l, c) in zip(sources1, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            #print(loc)
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        #print(loc.size())
        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                #self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                #self.priors
            )
        return output
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [BasicConv(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [BasicConv(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


extras = {
    '320': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}

def build_net(phase,size=320, num_classes=21):
    if size != 320 and size != 512:
        print("Error: Sorry only SSD300 and SSD512 is supported currently!")
        return
    
    return SSD(phase,add_extras(extras[str(size)],1024),num_classes=num_classes,size=size)


    
    
    
    
    
    
