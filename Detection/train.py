from __future__ import print_function
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
import argparse
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from data import *
from layers.modules import MultiBoxLoss_single_cross_riou_weight_margin
from layers.pixel_seg4 import downsample
from layers.functions import PriorBox,Detect
from utils.log import get_logger
import time

parser = argparse.ArgumentParser(
    description='FSSD Training')
parser.add_argument('-v', '--version', default='dssd320_fpn', help='version.')
parser.add_argument('-s', '--size', default='320', help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC', help='VOC or COCO dataset')
parser.add_argument('--basenet', default='resnet50-19c8e357.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=32,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument(
    '--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')
parser.add_argument('-max','--max_epoch', default=271,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='./weights/',
                    help='Location to save checkpoint models')
parser.add_argument('--save_val_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('-wu','--warm_epoch', default='1', type=int, help='warm up')
parser.add_argument('-txt_name', default='dssd_gai2_single_cross_weight_margin_riou', type=str, help='txt_name')
args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'VOC':
    train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    cfg = (voc320, voc512)[args.size == '512']
else:
    train_sets = [('2017', 'train')]
    cfg = (coco320, COCO_512)[args.size == '512']

if args.version == 'FSSD_VGG':
    from models.FSSD_VGG import build_net
elif args.version == 'FSSD_VGG_BN':
    from models.FSSD_VGG_BN import build_net
elif args.version == 'FSSD_VGG_prune':
    from models.FSSD_VGG_prune import build_net
elif args.version=='dssd320_fpn':
    from ssd_resnet.resnet_dssd import build_net
else:
    print('Unkown version!')

img_dim = (320,512)[args.size=='512']
rgb_means = (104, 117, 123)
p = 0.6
num_classes = (21, 81)[args.dataset == 'COCO']
batch_size = args.batch_size
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9
print(img_dim)
print(cfg)
print(num_classes)
dssd_net = build_net('train',img_dim, num_classes)

print(dssd_net)
logger = get_logger('./txt/'+args.txt_name+'.log')
if not args.resume_net:

    vgg_weights = torch.load(args.save_folder + args.basenet)
    print('Loading base network...')
    dssd_net.load_state_dict(vgg_weights,strict=False)

    def xavier(param):
        init.xavier_uniform(param)

    def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0

    print('Initializing weights...')
    # initialize newly added layers' weights with kaiming_normal method
    dssd_net.extras.apply(weights_init)
    dssd_net.loc.apply(weights_init)
    dssd_net.conf.apply(weights_init)

else:
    print('Loading resume network')
    print(args.resume_net)
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    dssd_net.load_state_dict(new_state_dict)

if args.ngpu > 1:
    dssd_net = torch.nn.DataParallel(dssd_net, device_ids=list(range(args.ngpu)))

if args.cuda:
    dssd_net.cuda()
    cudnn.benchmark = True

optimizer = optim.SGD(dssd_net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)

criterion = MultiBoxLoss_single_cross_riou_weight_margin(num_classes, 0.5, True, 0, True, 3, 0.5, False)
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()
        
def updateBN(s=0.0001):
    for m in dssd_net.modules():
        if isinstance(m,torch.nn.BatchNorm2d):
            m.weight.grad.detach().add_(s*torch.sign(m.weight.detach()))
        
def train():
    dssd_net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    loss_all=0
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    if args.dataset == 'VOC':
        dataset = VOCDetection(VOCroot, train_sets, preproc(
            img_dim, rgb_means, p), AnnotationTransform())
    elif args.dataset == 'COCO':
        dataset = COCODetection(COCOroot, train_sets, preproc(
            img_dim, rgb_means, p))
    else:
        print('Only VOC and COCO are supported now!')
        return

    epoch_size = len(dataset) // args.batch_size
    max_iter = args.max_epoch * epoch_size

    stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
    stepvalues_COCO = (90 * epoch_size, 130 * epoch_size, 150 * epoch_size)
    stepvalues = (stepvalues_VOC,stepvalues_COCO)[args.dataset=='COCO']
    print('Training',args.version, 'on', dataset.name)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
        for sv in stepvalues:
            if start_iter>sv:
                step_index+=1
                continue
            else:
                break
    else:
        start_iter = 0

    lr = args.lr
    avg_loss_list = []
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate))
            avg_loss = (loss_all)/epoch_size
            avg_loss_list.append(avg_loss)
            print("avg_loss_list:")
            if len(avg_loss_list)<=5:
                print (avg_loss_list)
            else:
                print(avg_loss_list[-5:])
            loc_loss = 0
            conf_loss = 0
            loss_all=0
            if (epoch % 5 == 0):
                torch.save(dssd_net.state_dict(), args.save_folder+'dssd_voc320/dssd_gai2_single_cross_weight_margin_riou/'+args.version+'_'+args.dataset + '_epoches_'+
                           repr(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda()) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno) for anno in targets]
        out =dssd_net(images)
        optimizer.zero_grad()
        loss_loc1, loss_c,loss_c_pos,loss_cross,loss_dis_weight_margin,loss_riou = criterion(out, priors, targets)
        loss = loss_cross+loss_c-loss_c_pos+loss_dis_weight_margin

        loss.backward()
        optimizer.step()
        t1 = time.time()

        loss_all+=loss.item()
        load_t1 = time.time()
        if iteration % 10 == 0:
            logger.info('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L: %.4f C: %.4f ||' % (
                loss_loc1.item(),loss_c.item())+'S: %.4f||'%(loss.item()) +'loss_c_pos: %.4f ||' % (loss_c_pos.item())
                 +'loss_cross_relation: %.4f ||' % (loss_cross.item())
                 +'loss_dis_weight_margin: %.4f ||' % (loss_dis_weight_margin.item())+'loss_riou: %.4f ||' % (loss_riou.item())
                  +'Batch time: %.4f ||' % (load_t1 - load_t0) + 'LR: %.7f' % (lr))

    torch.save(dssd_net.state_dict(), args.save_folder +
               'Final_' + args.version +'_' + args.dataset+ '.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < args.warm_epoch:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * 5)
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


if __name__ == '__main__':
    train()
