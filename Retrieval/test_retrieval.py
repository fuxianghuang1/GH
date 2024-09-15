# Copyright 2018 Google Inc. All Rights Reserved.
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

"""Evaluates the retrieval model."""
import numpy as np
import torch
from tqdm import tqdm as tqdm
from torchvision import utils as vutils
from torch.autograd import Variable
      
def get_imgadv(opt, data, model, flag_target=False):
   eps = opt.eps
   model.train()
   trainloader = data.get_loader(
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=opt.loader_num_workers)

   for data in trainloader:
    assert type(data) is list
    img1 = np.stack([d['source_img_data'] for d in data])
    img1 = torch.from_numpy(img1).float()
    img1 = torch.autograd.Variable(img1, requires_grad=True).cuda()
    img2 = np.stack([d['target_img_data'] for d in data])
    img2 = torch.from_numpy(img2).float()
    img2 = torch.autograd.Variable(img2, requires_grad=True).cuda()
    mods = [str(d['mod']['str']) for d in data]
    mods = [t.decode('utf-8') for t in mods]
    vutils.save_image(img1, 'img1.jpg')
      # compute loss

    if opt.loss == 'soft_triplet':
        loss_value = model.compute_loss(
            img1, mods, img2, soft_triplet_loss=True)
    elif opt.loss == 'batch_based_classification':
        loss_value = model.compute_loss(
            img1, mods, img2, soft_triplet_loss=False)
    else:
        print 'Invalid loss function', opt.loss
        sys.exit()

    if flag_target:
        loss = -loss_value
    else:
        loss = loss_value 
    
    model.zero_grad()
    img1.retain_grad()
    loss.backward()
    print 'grad:', type(img1)
    grad = img1.grad.cpu().detach().numpy()
    
    grad = np.sign(grad) 

    pertubation = grad * eps
    adv_img = img1.cpu().detach().numpy() + pertubation
    adv_img = torch.from_numpy(adv_img)
    pertubation = torch.from_numpy(pertubation)
    
    vutils.save_image(adv_img, 'adv_img.jpg')
    vutils.save_image(pertubation, 'pertubation.jpg')
    return adv_img, pertubation
    
def test_adv(opt, model, testset,flag_target=True):
  """Tests a model over the given testset."""
  model_test = model
  model_test.eval()
  test_queries = testset.get_test_queries()
  eps = opt.eps
  clip_min = 0.0
  clip_max = 1.0
  all_imgs = []
  all_captions = []
  all_queries = []
  all_target_captions = []
  if test_queries:
    # compute test query features
    imgs = []
    imgs2 = []
    mods = []  
    model_adv = model
    model_adv.train() 
    for t in tqdm(test_queries):
      imgs += [testset.get_img(t['source_img_id'])]
      #print 'imgs_type:', type(imgs)
      imgs2 += [testset.get_img(t['target_img_id'])]
      mods += [t['mod']['str']]
      if len(imgs) >= opt.batch_size or t is test_queries[-1]:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]#from numpy to torch
          imgs2 = [torch.from_numpy(d).float() for d in imgs2]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs, requires_grad=True).cuda()
        imgs2 = torch.stack(imgs2).float()
        imgs2 = torch.autograd.Variable(imgs2, requires_grad=True).cuda()        
        mods = [t.decode('utf-8') for t in mods]
        # add  pertubation   
        if opt.loss == 'soft_triplet':
           loss_value = model_adv.compute_loss(imgs, mods, imgs2, soft_triplet_loss=True)
        elif opt.loss == 'batch_based_classification':
           loss_value = model_adv.compute_loss(imgs, mods, imgs2, soft_triplet_loss=False)
        else:
           print 'Invalid loss function', opt.loss
           sys.exit()

        if flag_target:
           loss = -loss_value
        else:
           loss = loss_value 
    
        model_adv.zero_grad()
        imgs.retain_grad()
        loss.backward()
        grad = imgs.grad.cpu().detach().numpy()    
        grad = np.sign(grad) 

        pertubation = grad * eps
        adv_img = imgs.cpu().detach().numpy() + pertubation
        adv_img=np.clip(adv_img,clip_min,clip_max)#clip to (0 ,1)
        adv_img = torch.from_numpy(adv_img).cuda()
        #pertubation = torch.from_numpy(pertubation)
        #print 'imgs_type:', type(imgs)
        #print 'adv_img_type:', type(adv_img)
        f = model_test.compose_img_text(adv_img, mods).data.cpu().numpy()
        #f = model_test.compose_img_text(imgs, mods).data.cpu().numpy()
        all_queries += [f]
        imgs = []
        imgs2 = []
        mods = []
    all_queries = np.concatenate(all_queries)#the feature of all queries
    all_target_captions = [t['target_caption'] for t in test_queries]
    
    
    # compute all image features
    imgs = []
    for i in tqdm(range(len(testset.imgs))):
      try:    
        imgs += [testset.get_img(i)]
      except EOFError: 
        print i     
      if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs).cuda()
        imgs = model_test.extract_img_feature(imgs).data.cpu().numpy()
        #print 'image:', imgs
        #print 'imageshape:', np.array(imgs).shape #--output:imageshape:(32,512) (batch_size,embed_dim)
        all_imgs += [imgs]
        #print 'allimageshape:', np.array(all_imgs).shape  #--output:allimageshape:(n,32,512)
       
        imgs = []
    all_imgs = np.concatenate(all_imgs)
    all_captions = [img['captions'][0] for img in testset.imgs]
    #print 'all_captions:', all_captions
    print 'all_captions:', np.array(all_captions).shape
    #print 'allimageshape:', np.array(all_imgs).shape #--output:allimageshape:(19012,512) (trainsetsize,embed_dim)
    #print 'all_captions:', np.array(all_captions).shape #--output:all_captions:(19012,) (trainsetsize,)
  else:
    # use training queries to approximate training retrieval performance
    imgs0 = []
    imgs = []
    mods = []
    for i in range(10000):
      item = testset[i]
      imgs += [item['source_img_data']]
      mods += [item['mod']['str']]
      if len(imgs) > opt.batch_size or i == 9999:
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)
        mods = [t.decode('utf-8') for t in mods] #utf-8 to unicode
        f = model_test.compose_img_text(imgs.cuda(), mods).data.cpu().numpy()
        all_queries += [f]
        imgs = []
        mods = []
      imgs0 += [item['target_img_data']]
      if len(imgs0) > opt.batch_size or i == 9999:
        imgs0 = torch.stack(imgs0).float()
        imgs0 = torch.autograd.Variable(imgs0)
        imgs0 = model_test.extract_img_feature(imgs0.cuda()).data.cpu().numpy()
        all_imgs += [imgs0]
        imgs0 = []
      all_captions += [item['target_caption']]
      all_target_captions += [item['target_caption']]
    all_imgs = np.concatenate(all_imgs)
    all_queries = np.concatenate(all_queries)

  # feature normalization
  for i in range(all_queries.shape[0]):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  for i in range(all_imgs.shape[0]):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  sims = all_queries.dot(all_imgs.T)
  #print 'sims:', sims
  
  if test_queries:
    for i, t in enumerate(test_queries):
      sims[i, t['source_img_id']] = -10e10  # remove query image
  nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]#return top 110 index
  #print 'nn_result0:', nn_result[0]
  #print 'nn_result0size:', np.array(nn_result).shape 

  # compute recalls
  out = []
  nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
  #print 'nn_result:', nn_result[0]
  #print 'nn_resultsize:', np.array(nn_result).shape #--output:nn_resultsize:(10000,110) 
  
  for k in [1, 5, 10, 50, 100]:
    r = 0.0
    for i, nns in enumerate(nn_result):
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(nn_result)
    out += [('recall_top' + str(k) + '_correct_composition', r)]

    if opt.dataset == 'mitstates':
      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[0] in [c.split()[0] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_adj', r)]

      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[1] in [c.split()[1] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_noun', r)]

  return out
  
  
def test(opt, model, testset):
  """Tests a model over the given testset."""
  model.eval()
  test_queries = testset.get_test_queries()

  all_imgs = []
  all_captions = []
  all_queries = []
  all_target_captions = []
  if test_queries:
    # compute test query features
    imgs = []
    mods = []
    for t in tqdm(test_queries):
      imgs += [testset.get_img(t['source_img_id'])]
      mods += [t['mod']['str']]
      if len(imgs) >= opt.batch_size or t is test_queries[-1]:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs).cuda()
        mods = [t.decode('utf-8') for t in mods]
        f = model.compose_img_text(imgs, mods).data.cpu().numpy()
        all_queries += [f]
        imgs = []
        mods = []
    all_queries = np.concatenate(all_queries)
    all_target_captions = [t['target_caption'] for t in test_queries]

    # compute all image features
    imgs = []
    for i in tqdm(range(len(testset.imgs))):
      imgs += [testset.get_img(i)]
      if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs).cuda()
        imgs = model.extract_img_feature(imgs).data.cpu().numpy()
        all_imgs += [imgs]
        imgs = []
    all_imgs = np.concatenate(all_imgs)
    all_captions = [img['captions'][0] for img in testset.imgs]

  else:
    # use training queries to approximate training retrieval performance
    imgs0 = []
    imgs = []
    mods = []
    for i in range(10000):
      item = testset[i]
      imgs += [item['source_img_data']]
      mods += [item['mod']['str']]
      if len(imgs) > opt.batch_size or i == 9999:
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)
        mods = [t.decode('utf-8') for t in mods]
        f = model.compose_img_text(imgs.cuda(), mods).data.cpu().numpy()
        all_queries += [f]
        imgs = []
        mods = []
      imgs0 += [item['target_img_data']]
      if len(imgs0) > opt.batch_size or i == 9999:
        imgs0 = torch.stack(imgs0).float()
        imgs0 = torch.autograd.Variable(imgs0)
        imgs0 = model.extract_img_feature(imgs0.cuda()).data.cpu().numpy()
        all_imgs += [imgs0]
        imgs0 = []
      all_captions += [item['target_caption']]
      all_target_captions += [item['target_caption']]
    all_imgs = np.concatenate(all_imgs)
    all_queries = np.concatenate(all_queries)

  # feature normalization
  for i in range(all_queries.shape[0]):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  for i in range(all_imgs.shape[0]):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  sims = all_queries.dot(all_imgs.T)
  if test_queries:
    for i, t in enumerate(test_queries):
      sims[i, t['source_img_id']] = -10e10  # remove query image
  nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]

  # compute recalls
  out = []
  nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
  for k in [1, 5, 10, 50, 100]:
    r = 0.0
    for i, nns in enumerate(nn_result):
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(nn_result)
    out += [('recall_top' + str(k) + '_correct_composition', r)]
    

    '''if opt.dataset == 'mitstates':
      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[0] in [c.split()[0] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_adj', r)]

      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[1] in [c.split()[1] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_noun', r)]'''
    

  return out
