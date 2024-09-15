import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from modeling.resnet_base import BasicBlock,Bottleneck
#from data import voc, coco
import torchvision
import os
import math


#extra layers
'''extras = {
    'resnet50': [1024,256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    'resnet101': [512,256,128,64],
}

#where to extract features
extract = {
    'resnet50': {'b':[21,33],'e':[1,3,5,7]}, #vgg -14
    'resnet101': {'b':[10,16,19],'e':[0,1,2]}
}


mbox = {
    'vgg': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    'resnet': [4, 6, 6, 6, 4, 4],
}'''




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

class resnet_ssd_fpn(nn.Module):
    def __init__(self, phase,extras,size,block, layers, num_classes=1000):
        super(resnet_ssd_fpn, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        #self.cfg = (coco, voc)[num_classes == 21]
        #self.priorbox = PriorBox(self.cfg)
        #self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size
        self.inplanes = 64
        #widths = [int(round(ch * width)) for ch in [64, 128, 256, 512]]
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        #print(layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # change stride = 2, dilation = 1 in ResNet to stride = 1, dilation = 2 for the final _make_layer
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,dilation=2)
        
        # remove the final avgpool and fc layers
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(widths[3] * block.expansion, num_classes)
        # add extra layers
        self.extras = nn.ModuleList(extras)
        #print(".......")
        #print(self.extras.Conv2d)
        '''self.deconv1=BasicdeConv(512, 512, kernel_size=3, stride=1, padding=0, relu=False,bn=False)
        self.deconv2=BasicdeConv(512, 1024, kernel_size=3, stride=1, padding=0, relu=False,bn=False)
        self.deconv3=BasicdeConv(1024, 1024, kernel_size=2, stride=2, padding=0, relu=False,bn=False)
        self.deconv4=BasicdeConv(1024, 2048, kernel_size=2, stride=2, padding=0,relu=False,bn=False)
        self.deconv5=BasicdeConv(2048, 512, kernel_size=2, stride=2, padding=0, relu=False,bn=False)'''

        self.latlayer1=nn.ModuleList([BasicConv(512,512,kernel_size=3,stride=1,padding=1,relu=False,bn=True),
                                      BasicConv(1024,512,kernel_size=3,stride=1,padding=1,relu=False,bn=True),
                                      BasicConv(1024,512,kernel_size=3,stride=1,padding=1,relu=False,bn=True),
                                      BasicConv(1024,512,kernel_size=3,stride=1,padding=1,relu=False,bn=True),
                                      BasicConv(2048,512,kernel_size=3,stride=1,padding=1,relu=False,bn=True),
                                      BasicConv(512,512,kernel_size=3,stride=1,padding=1,relu=False,bn=True)])
        
        self.smooth=nn.ModuleList([   BasicConv(512,512,kernel_size=3,stride=1,padding=1,relu=False,bn=True),
                                      BasicConv(512,512,kernel_size=3,stride=1,padding=1,relu=False,bn=True),
                                      BasicConv(512,512,kernel_size=3,stride=1,padding=1,relu=False,bn=True),
                                      BasicConv(512,512,kernel_size=3,stride=1,padding=1,relu=False,bn=True),
                                      BasicConv(512,512,kernel_size=3,stride=1,padding=1,relu=False,bn=True),
                                      BasicConv(512,512,kernel_size=3,stride=1,padding=1,relu=False,bn=True)])
        '''self.latlayer2=nn.ModuleList([BasicConv(512,512,kernel_size=3,stride=1,padding=1,relu=False,bn=True),
                                      BasicConv(1024,1024,kernel_size=3,stride=1,padding=1,relu=False,bn=True),
                                      BasicConv(1024,1024,kernel_size=3,stride=1,padding=1,relu=False,bn=True),
                                      BasicConv(2048,2048,kernel_size=3,stride=1,padding=1,relu=False,bn=True),
                                      BasicConv(512,512,kernel_size=3,stride=1,padding=1,relu=False,bn=True)])'''
        '''self.latlayer1=BasicConv(512,512,kernel_size=3,stride=1,padding=1,relu=False,bn=True)
        self.latlayer2=BasicConv(1024,1024,kernel_size=3,stride=1,padding=1,relu=False,bn=True)
        self.latlayer3=BasicConv(1024,1024,kernel_size=3,stride=1,padding=1,relu=False,bn=True)
        self.latlayer4=BasicConv(2048,2048,kernel_size=3,stride=1,padding=1,relu=False,bn=True)
        self.latlayer5=BasicConv(512,512,kernel_size=3,stride=1,padding=1,relu=False,bn=True)'''
        self.loc=nn.ModuleList([nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(512, 24, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(512, 24, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(512, 16, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(512, 16, kernel_size=3, stride=1, padding=1), \
                                
                                ])
        self.conf=nn.ModuleList([nn.Conv2d(512, num_classes*8, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(512, num_classes*8, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(512, num_classes*8, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(512, num_classes*6, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(512, num_classes*6, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(512, num_classes*4, kernel_size=3, stride=1, padding=1), \
                                nn.Conv2d(512, num_classes*4, kernel_size=3, stride=1, padding=1), \
                               ])
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            #self.detect = Detect(num_classes, 0, 400, 0.01, 0.45)
        '''for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.bias.data.zero_()
                #print(m.weight.data[0])
            
                #nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()'''

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            #print("true")
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=dilation, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y
    
    def forward(self, x):
        sources1 = list()#origin ssd sources
        sources2 = list()#deconv layers
        sources=list()#predction layers
        loc = list()
        conf = list()
        mask1=list()
        mask2=list()
        mask3=list()
        mask4=list()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        sources1.append (x)
        x = self.layer3(x)
        x = self.layer4(x)
        sources1.append (x)
        for k,v in  enumerate(self.extras):
            x=v(x)
            if k % 2 == 1:
                #print(v)
                #print(x.shape)
                sources1.append(x)
        p1=sources1[-1]
        sources.append(p1)
        p2=self._upsample_add(p1,self.latlayer1[0](sources1[-2]))
        sp2=self.smooth[0](p2)
        pred2=F.relu(sp2,inplace=True)
        sources.append(pred2)
        
        p3=self._upsample_add(p2,self.latlayer1[1](sources1[-3]))
        sp3=self.smooth[1](p3)
        pred3=F.relu(sp3,inplace=True)
        sources.append(pred3)

        p4=self._upsample_add(p3,self.latlayer1[2](sources1[-4]))
        sp4=self.smooth[2](p4)
        pred4=F.relu(sp4,inplace=True)
        sources.append(pred4)

        p5=self._upsample_add(p4,self.latlayer1[3](sources1[-5]))
        sp5=self.smooth[3](p5)
        pred5=F.relu(sp5,inplace=True)
        sources.append(pred5)

        p6=self._upsample_add(p5,self.latlayer1[4](sources1[-6]))
        sp6=self.smooth[4](p6)
        pred6=F.relu(sp6,inplace=True)
        sources.append(pred6)  
    
        p7=self._upsample_add(p6,self.latlayer1[5](sources1[-7]))
        sp7=self.smooth[5](p7)
        pred7=F.relu(sp7,inplace=True)
        sources.append(pred7)                          
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
        for (x, l, c) in zip(sources[::-1], self.loc, self.conf):
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
        #out10x10, out5x5, out3x3, out1x1 = self.extra_layers(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)


def make_layer2(block, inplanes, planes, blocks, stride=1,delation=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers = []
    layers.append(block(inplanes,planes,stride,downsample,delation))  
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))
    return layers



def resnet_extras(block):        
    ex = []
    #in_channels=0
    ex+=make_layer2(block,2048,256,2,stride=2)
    ex+=make_layer2(block,1024,256,2,stride=2)
    ex+=make_layer2(block,1024,256,2,stride=2)
    ex+=make_layer2(block,1024,128,2,stride=2)
    ex+=make_layer2(block,512,128,2,stride=2)
    '''ex+= [BasicConv(2048, 512, kernel_size=1,stride=1)]
    ex+=[BasicConv(512, 1024, kernel_size=3,padding=1,stride=2)]
    ex+= [BasicConv(1024, 512, kernel_size=1,stride=1)]
    ex+=[BasicConv(512, 1024, kernel_size=3,padding=1,stride=2)]
    ex+= [BasicConv(1024, 256, kernel_size=1,stride=1)]
    ex+=[BasicConv(256, 512, kernel_size=3,stride=1)]'''
    #ex+= [BasicConv(512, 256, kernel_size=1,stride=1)]
    #ex+=[BasicConv(256, 512, kernel_size=3,stride=1)]
    return ex

def build_net(phase, size=512, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300 and size!=512:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    block=Bottleneck
    extras_=resnet_extras(block)
    return resnet_ssd_fpn(phase,extras_,size,block,[3,4,6,3],num_classes=num_classes)





