# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Main method to train the model."""

#!/usr/bin/python

import argparse
import sys
import time
import datasets
import img_text_composition_models
import numpy as np
from tensorboardX import SummaryWriter
import test_retrieval
import torch
import torch.utils.data
import torchvision
from tqdm import tqdm as tqdm
import math

torch.set_num_threads(3)
torch.cuda.set_device(1)



def parse_opt():
  """Parses the input arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', type=str, default='')
  parser.add_argument('--comment', type=str, default='css3d')
  parser.add_argument('--dataset', type=str, default='css3d')
  parser.add_argument(
      '--dataset_path', type=str, default='./data/css3d')
  parser.add_argument('--model', type=str, default='tirg_lastconv')
  parser.add_argument('--embed_dim', type=int, default=512)
  parser.add_argument('--learning_rate', type=float, default=1e-2)
  parser.add_argument(
      '--learning_rate_decay_frequency', type=int, default=50000)
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--weight_decay', type=float, default=1e-6)
  parser.add_argument('--num_iters', type=int, default=160000)
  parser.add_argument('--epoch', type=int, default=600)
  parser.add_argument('--loss', type=str, default='soft_triplet')
  parser.add_argument('--loader_num_workers', type=int, default=4)
  parser.add_argument('--test_only', type=bool, default=True)  
  parser.add_argument('--test_adv', type=bool, default=False)
  parser.add_argument('--GH', type=bool, default=False, help="utilize GH or not)")
  parser.add_argument('--GH_new', type=bool, default=True, help="utilize GH++ or not)")
  parser.add_argument('--eps', type=float, default=0.3)
  parser.add_argument('--model_checkpoint', type=str, default=' ')
  parser.add_argument('--pretrained_model_checkpoint', type=str, default=' ')
  return args


def load_dataset(opt):
  """Loads the input datasets."""
  print ('Reading dataset ', opt.dataset)
  if opt.dataset == 'css3d':
    trainset = datasets.CSSDataset(
        path=opt.dataset_path,
        #image_resize=opt.image_resize,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.CSSDataset(
        path=opt.dataset_path,
        #image_resize=opt.image_resize,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
  elif opt.dataset == 'fashion200k':
    trainset = datasets.Fashion200k(
        path=opt.dataset_path,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.Fashion200k(
        path=opt.dataset_path,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
  elif opt.dataset == 'mitstates':
    trainset = datasets.MITStates(
        path=opt.dataset_path,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.MITStates(
        path=opt.dataset_path,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
  else:
    print ('Invalid dataset', opt.dataset)
    sys.exit()

  return trainset, testset


def create_model_and_optimizer(opt, texts):
  """Builds the model and related optimizer."""
  print ('Creating model and optimizer for', opt.model)
  if opt.model == 'imgonly':
    model = img_text_composition_models.SimpleModelImageOnly(
        texts, embed_dim=opt.embed_dim)
  elif opt.model == 'textonly':
    model = img_text_composition_models.SimpleModelTextOnly(
        texts, embed_dim=opt.embed_dim)
  elif opt.model == 'concat':
    model = img_text_composition_models.Concat(texts, embed_dim=opt.embed_dim)
  elif opt.model == 'tirg':
    model = img_text_composition_models.TIRG(texts, embed_dim=opt.embed_dim)
  elif opt.model == 'tirg_lastconv':
    model = img_text_composition_models.TIRGLastConv(
        texts, embed_dim=opt.embed_dim)
  else:
    print ('Invalid model', opt.model)
    print ('available: imgonly, textonly, concat, tirg or tirg_lastconv')
    sys.exit()
  model = model.cuda()

  # create optimizer
  params = []
  # low learning rate for pretrained layers on real image datasets
  if opt.dataset != 'css3d':
    params.append({
        'params': [p for p in model.img_model.fc.parameters()],
        'lr': opt.learning_rate
    })
    params.append({
        'params': [p for p in model.img_model.parameters()],
        'lr': 0.1 * opt.learning_rate
    })
  params.append({'params': [p for p in model.parameters()]})
  for _, p1 in enumerate(params):  # remove duplicated params
    for _, p2 in enumerate(params):
      if p1 is not p2:
        for p11 in p1['params']:
          for j, p22 in enumerate(p2['params']):
            if p11 is p22:
              p2['params'][j] = torch.tensor(0.0, requires_grad=True)
  optimizer = torch.optim.SGD(
      params, lr=opt.learning_rate, momentum=0.9, weight_decay=opt.weight_decay)
  return model, optimizer


def train_loop(opt, logger, trainset, testset, model, optimizer):
  """Function for train loop"""
  print ('Begin training')
  losses_tracking = {}
  it = 0
  epoch = -1
  tic = time.time()
  while  epoch < opt.epoch: #it < opt.num_iters:
    epoch += 1

    # show/log stats
    print ('It', it, 'epoch', epoch, 'Elapsed time', round(time.time() - tic,
                                                          4), opt.comment)
    tic = time.time()
    for loss_name in losses_tracking:
      avg_loss = np.mean(losses_tracking[loss_name][-len(trainloader):])
      print ('    Loss', loss_name, round(avg_loss, 4)  )  
      logger.add_scalar(loss_name, avg_loss, it)
    logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], it)

    # test
    if epoch % 5 == 0 and epoch > 1:
      tests = []
      tests_adv = []
      for name, dataset in [('train', trainset), ('test', testset)]:
          t = test_retrieval.test(opt, model, dataset)    
          tests += [(name + ' ' + metric_name, metric_value) for metric_name, metric_value in t]    
      for metric_name, metric_value in tests:
          logger.add_scalar(metric_name, metric_value, epoch)
          print('    org:', metric_name, round(metric_value, 4))
            
      if opt.test_adv:
          for name, dataset in [('train', trainset), ('test', testset)]:
            t_adv = test_retrieval.test_adv(opt, model, dataset) 
            tests_adv += [(name + ' ' + metric_name, metric_value) for metric_name, metric_value in t_adv]  
          for metric_name, metric_value in tests_adv:
            logger.add_scalar(metric_name, metric_value, epoch)
            print ('    adv:', metric_name, round(metric_value, 4))
    # save checkpoint
      torch.save({
          'epoch': epoch,
          'opt': opt,
          'model_state_dict': model.state_dict(),
      },
                 logger.file_writer.get_logdir() + '/epoch'+ str(epoch)+ '_checkpoint.pth')
 
    # run trainning for 1 epoch
    model.train()
    trainloader = trainset.get_loader(
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=opt.loader_num_workers)


    def training_1_iter(data):
      assert type(data) is list
      img1 = np.stack([d['source_img_data'] for d in data])
      img1 = torch.from_numpy(img1).float()
      img1 = torch.autograd.Variable(img1, requires_grad=True).cuda()
      img2 = np.stack([d['target_img_data'] for d in data])
      img2 = torch.from_numpy(img2).float()
      img2 = torch.autograd.Variable(img2, requires_grad=True).cuda()
      mods = [str(d['mod']['str']) for d in data]
      mods = [t.decode('utf-8') for t in mods]
      

      pertubation = torch.randn(img1.shape)
      adv_img = (img1 + opt.eps * pertubation.cuda()).cpu().detach().numpy() 
      adv_img =np.clip(adv_img,0.0,1.0)
      adv_img = torch.from_numpy(adv_img).cuda()
      # compute loss
      losses = []
      #alpha = 0.5
      if opt.loss == 'soft_triplet':
        loss_org = model.compute_loss(img1, mods, img2, soft_triplet_loss=True)
        loss_adv = model.compute_loss(adv_img, mods, img2, soft_triplet_loss=True)
      elif opt.loss == 'batch_based_classification':
        loss_org = model.compute_loss(img1, mods, img2, soft_triplet_loss=False)
        loss_adv = model.compute_loss(adv_img, mods, img2, soft_triplet_loss=False)
      else:
        print ('Invalid loss function', opt.loss)
        sys.exit()
      if opt.GH: 
        loss_value = harmonicgradloss_GH(loss_1=loss_org.cuda(), loss_2=loss_adv.cuda(), net=model, optimizer=optimizer) 
      elif opt.GH_new:
        loss_value = harmonicgradloss_GH_new(loss_1=loss_org.cuda(), loss_2=loss_adv.cuda(), net=model, optimizer=optimizer)  
      else:
        loss_value = loss_org + loss_adv
      loss_name = opt.loss
      loss_weight = 1.0
      losses += [(loss_name, loss_weight, loss_value, loss_org, loss_adv)]
      total_loss = sum([
          loss_weight * loss_value
          for loss_name, loss_weight, loss_value, loss_org, loss_adv in losses
      ])
      #assert not torch.isnan(total_loss)
      #losses += [('total training loss', None, total_loss, loss_org, loss_adv)]

      # track losses
      for loss_name, loss_weight, loss_value, loss_org, loss_adv in losses:
        if not losses_tracking.has_key(loss_name):
          losses_tracking[loss_name] = []
        losses_tracking[loss_name].append(float(loss_value))
        if not losses_tracking.has_key('loss_org'):
          losses_tracking['loss_org'] = []
        losses_tracking['loss_org'].append(float(loss_org))
        if not losses_tracking.has_key('loss_adv'):
          losses_tracking['loss_adv'] = []
        losses_tracking['loss_adv'].append(float(loss_adv))

      # gradient descend
      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()

    #for data in tqdm(trainloader, desc='Training for epoch ' + str(epoch)):
    for data in trainloader:
      it += 1
      training_1_iter(data)

      # decay learing rate
      if it >= opt.learning_rate_decay_frequency and it % opt.learning_rate_decay_frequency == 0:
        for g in optimizer.param_groups:
          g['lr'] *= 0.1

  print ('Finished training')

def get_imgadv(opt, img1, mods, img2, model, flag_target=False):
    #model.train()
    eps = opt.eps  
    clip_min = 0.0
    clip_max = 1.0
   
      # compute loss

    if opt.loss == 'soft_triplet':
        loss_value = model.compute_loss(
            img1, mods, img2, soft_triplet_loss=True)
    elif opt.loss == 'batch_based_classification':
        loss_value = model.compute_loss(
            img1, mods, img2, soft_triplet_loss=False)
    else:
        print ('Invalid loss function', opt.loss)
        sys.exit()

    if flag_target:
        loss = -loss_value
    else:
        loss = loss_value 
    
    model.zero_grad()
    img1.retain_grad()
    loss.backward()
    grad = img1.grad.cpu().detach().numpy()
    grad = np.sign(grad) 

    pertubation = grad * eps
    adv_img = img1.cpu().detach().numpy() + pertubation
    adv_img=np.clip(adv_img,clip_min,clip_max)#clip to (0 ,1)
    adv_img = torch.from_numpy(adv_img).cuda()
    pertubation = torch.from_numpy(pertubation)

    return adv_img, pertubation

def getgrad(net):
    g=[]
    for name, param in net.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                g.append(torch.tensor(param.grad).view(-1).contiguous())
    g = torch.cat(g, dim=0).detach()
    return g

def harmonicgradloss_GH(loss_1, loss_2, net, optimizer):
    optimizer.zero_grad()
    loss_1.backward(retain_graph=True)
    grad1 = getgrad(net)
    optimizer.zero_grad()
    loss_2.backward(retain_graph=True)
    grad2 = getgrad(net)
    optimizer.zero_grad()
    if torch.dot(grad1,grad2) < 0:
        gh1 = 1 - (torch.dot(grad1,grad2))/(torch.dot(grad1,grad1))
        gh2 = 1 - (torch.dot(grad1,grad2))/(torch.dot(grad2,grad2))
    else:
        gh1 = 1
        gh2 = 1    
    total_loss =  gh1 * loss_1 + gh2 * loss_2 
    return total_loss

def harmonicgradloss_GH_new(loss_1, loss_2, net, optimizer):
    optimizer.zero_grad()
    loss_1.backward(retain_graph=True)
    grad1 = getgrad(net)
    optimizer.zero_grad()
    loss_2.backward(retain_graph=True)
    grad2 = getgrad(net)
    optimizer.zero_grad()
    dot_product = torch.dot(grad1, grad2)
    if dot_product < 0:            
        norm1 = torch.norm(grad1)
        norm2 = torch.norm(grad2)
        cosine_similarity = dot_product / (norm1 * norm2)
        angle_radians = torch.acos(cosine_similarity)
        angle_degrees = math.degrees(angle_radians.item()) #\theta
        #print('angle_degrees:', angle_degrees)
        theta = angle_degrees
        beta_max = angle_degrees - 90
        beta =  0.5 * beta_max
        gh1 = 1 + 2 * math.sin(math.radians(0.5 * beta))
        gh2 = 1 + 2 * math.sin(math.radians(0.5 *(beta + 90 - theta)))
    else:
        gh1 = 1
        gh2 = 1   
    total_loss =  gh1 * loss_1 + gh2 * loss_2
    return total_loss

    
def main():
  opt = parse_opt()
  print ('Arguments:')
  for k in opt.__dict__.keys():
    print ('    ', k, ':', str(opt.__dict__[k]))

  logger = SummaryWriter(comment=opt.comment)
  print ('Log files saved to', logger.file_writer.get_logdir())
  for k in opt.__dict__.keys():
    logger.add_text(k, str(opt.__dict__[k]))

  trainset, testset = load_dataset(opt)
  model, optimizer = create_model_and_optimizer(
      opt, [t.decode('utf-8') for t in trainset.get_all_texts()])

  if opt.test_only:
        print('Doing test only')
        checkpoint = torch.load(opt.model_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        model.eval()
        tests = []
        tests_adv = []
        epoch = 0
        for name, dataset in [('train', trainset), ('test', testset)]:
            t = test_retrieval.test(opt, model, dataset)    
            tests += [(name + ' ' + metric_name, metric_value) for metric_name, metric_value in t]    
        for metric_name, metric_value in tests:
            logger.add_scalar(metric_name, metric_value, epoch)
            print('    org:', metric_name, round(metric_value, 4))
            
        if opt.test_adv:
            for name, dataset in [('train', trainset), ('test', testset)]:
              t_adv = test_retrieval.test_adv(opt, model, dataset) 
              tests_adv += [(name + ' ' + metric_name, metric_value) for metric_name, metric_value in t_adv]  
            for metric_name, metric_value in tests_adv:
              logger.add_scalar(metric_name, metric_value, epoch)
              print ('    adv:', metric_name, round(metric_value, 4))
            
        return 0
  train_loop(opt, logger, trainset, testset, model, optimizer)
  logger.close()


if __name__ == '__main__':
  main()
