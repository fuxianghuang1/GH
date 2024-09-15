from __future__ import print_function
import torch.optim as optim
from model.build_gen import *
import numpy as np
# import ot
from basenet import *
import math
from utils import *
from torch.utils.data import DataLoader
from model import *
from tllib.utils.data import ForeverDataIterator
from tllib.utils.meter import AverageMeter, ProgressMeter

# Training settings
class Solver(object):
    def __init__(self, args, batch_size=128, learning_rate=0.0002, interval=100, optimizer='adam'
                 , num_k=4, all_use=False, checkpoint_dir=None, save_epoch=10):
        self.batch_size = batch_size
        self.iters = args.iters_per_epoch
        self.print_freq = args.print_freq
        self.data = args.data
        self.source = args.source
        self.target = args.target
        self.num_k = num_k
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.all_use = all_use
        #self.class_num= 10
        self.num_k1 = 8
        self.num_k2 = 1
        self.num_k3 = 8
        self.num_k4 = 1
        self.GH = args.GH
        self.GH_new = args.GH_new
        self.trade_off_1 = args.trade_off_1
        #self.offset =0.1
        self.output_cr_t_C_label = np.zeros(self.batch_size)

        if self.source[0] == 'SVHN':
            self.scale = True
        else:
            self.scale = False

        print('dataset loading')
        self.train_transform = get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                              random_horizontal_flip=not args.no_hflip,
                                              random_color_jitter=False, resize_size=args.resize_size,
                                              norm_mean=args.norm_mean, norm_std=args.norm_std)
        self.val_transform = get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                               norm_mean=args.norm_mean, norm_std=args.norm_std)

        self.train_source_dataset, self.train_target_dataset, self.val_dataset, self.test_dataset, self.num_classes, self.class_names = \
            get_dataset(self.data, args.root, self.source, self.target, self.train_transform, self.val_transform)
        self.train_source_loader = DataLoader(self.train_source_dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=2, drop_last=True)
        self.train_target_loader = DataLoader(self.train_target_dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=2, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        self.train_source_iter = ForeverDataIterator(self.train_source_loader)
        self.train_target_iter = ForeverDataIterator(self.train_target_loader)
        print('load finished!')

        if self.data == 'Digits':
            self.G = Generator(source=self.source[0], target=self.target[0])
            self.C = Classifier(source=self.source[0], target=self.target[0])
            self.C1 = Classifier(source=self.source[0], target=self.target[0])
            self.C2 = Classifier(source=self.source[0], target=self.target[0])
            self.D = discriminator(source=self.source[0], target=self.target[0])
        else:
            self.PoolLayer = PoolLayer()
            self.G = get_model(args.arch, pretrain=not args.scratch)
            self.C = Predictor(self.G.out_features, self.num_classes)
            self.C1 = Predictor(self.G.out_features, self.num_classes)
            self.C2 = Predictor(self.G.out_features, self.num_classes)
            self.D =AdversarialNetwork(self.G.out_features)
        if args.eval_only:
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source[0], self.target[0], args.resume_epoch))
            self.C.torch.load(
                '%s/%s_to_%s_model_epoch%s_C.pt' % (self.checkpoint_dir, self.source[0], self.target[0], args.resume_epoch))
            self.C1.torch.load(
                '%s/%s_to_%s_model_epoch%s_C1.pt' % (self.checkpoint_dir, self.source[0], self.target[0], args.resume_epoch))
            self.C2.torch.load(
                '%s/%s_to_%s_model_epoch%s_C2.pt' % (self.checkpoint_dir, self.source[0], self.target[0], args.resume_epoch))
            self.D.torch.load(
                '%s/%s_to_%s_model_epoch%s_D.pt' % (self.checkpoint_dir, self.source[0], self.target[0], args.resume_epoch))

        self.G.cuda()
        self.C.cuda()
        self.C1.cuda()
        self.C2.cuda()
        self.D.cuda()
        self.interval = interval

        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        self.lr = args.lr

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)
            self.opt_c = optim.SGD(self.C.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)
            self.opt_c1 = optim.SGD(self.C1.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)
            self.opt_c2 = optim.SGD(self.C2.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)
            self.opt_d = optim.SGD(self.D.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)

        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=lr, weight_decay=0.0005)
            self.opt_c = optim.Adam(self.C.parameters(),
                                    lr=lr, weight_decay=0.0005)
            self.opt_c1 = optim.Adam(self.C1.parameters(),
                                    lr=lr, weight_decay=0.0005)
            self.opt_c2 = optim.Adam(self.C2.parameters(),
                                    lr=lr, weight_decay=0.0005)
            self.opt_d = optim.Adam(self.D.parameters(),
                                    lr=lr, weight_decay=0.0005)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()
        self.opt_d.zero_grad()

    def getgrad(self, net):
        g = []
        for name, param in net.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    g.append(torch.as_tensor(param.grad).view(-1).contiguous())
        g = torch.cat(g, dim=0).detach()
        return g

    def props_to_onehot(self, props):
       props = props.cpu().detach().numpy()
       if isinstance(props, list):
          props = np.array(props)
       a = np.argmax(props, axis=1)
       b = np.zeros((len(a), props.shape[1]))
       b[np.arange(len(a)), a] = 1
       return torch.from_numpy(b)

    def ent(self, output):
        out = -torch.mean(F.softmax(output.cuda() + 1e-6)*torch.log(F.softmax(output.cuda() + 1e-6)))
        out = Variable(out,requires_grad=True)
        return out

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))
 
    def linear_mmd(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X - f_of_Y
        loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
        return loss

    def train(self, A_st_min, A_st_max, min_J_w, max_J_w, A_st_norm, J_w_norm, epoch, record_file=None):
    #---------- add for compute distance------#
        if self.data == 'Digits':
            if self.source[0] == 'USPS' and self.target[0] == 'MNIST':
                fea_for_LDA= np.empty(shape=(0, 768))
                fea_s_for_LDA = np.empty(shape=(0, 768))
                label_for_LDA = np.empty(shape=(0, 1))
                label_s_for_LDA = []
            if self.source[0] == 'MNIST' and self.target[0] == 'USPS':
                fea_for_LDA= np.empty(shape=(0, 768))
                fea_s_for_LDA = np.empty(shape=(0, 768))
                label_for_LDA = np.empty(shape=(0, 1))
                label_s_for_LDA = []
            if self.source[0] == 'SVHN':
                fea_for_LDA= np.empty(shape=(0, 3072))
                fea_s_for_LDA = np.empty(shape=(0, 3072))
                label_for_LDA = np.empty(shape=(0, 1))
                label_s_for_LDA = []
        else:
            fea_for_LDA = np.empty(shape=(0, 2048))
            fea_s_for_LDA = np.empty(shape=(0, 2048))
            label_for_LDA = np.empty(shape=(0, 1))
            label_s_for_LDA = []
    #-------------------end----------------#

        criterion = nn.CrossEntropyLoss().cuda()
        adv_loss = nn.BCEWithLogitsLoss().cuda()
        self.G.train()
        self.C.train()
        self.C1.train()
        self.C2.train()
        self.D.train()
        torch.cuda.manual_seed(1)
        loss_mmd_all = 0
        counter = 0

        for iter_idx in range(self.iters):
            if iter_idx % self.print_freq == 0:
                print('Epoch: [{}] [{}/{}]'.format(epoch, iter_idx, self.iters))
            img_s, label_s = next(self.train_source_iter)[:2]
            img_t, = next(self.train_target_iter)[:1]

            img_s = img_s.cuda()
            img_t = img_t.cuda()
            imgs = Variable(torch.cat((img_s, \
                                       img_t), 0))
            label_s = Variable(label_s.long().cuda())
            img_s = Variable(img_s)
            img_t = Variable(img_t)
            self.reset_grad()

##########################compute the T #############
            T_complex = A_st_norm/(A_st_norm+(1.0-J_w_norm))
            T = T_complex.real

########### source domain is discriminative.#################
            for i1 in range(self.num_k1):
                feat_cr_s = self.G(img_s)
                output_cr_s_C = self.C(feat_cr_s.cuda())
                loss_cr_s = criterion(output_cr_s_C, label_s) 
            
                loss_1 = loss_cr_s 
                loss_1.backward()
                self.opt_g.step()
                self.opt_c.step()
                self.reset_grad()

#################### transferability ########################
            for i2 in range(self.num_k2):
                feat_cr_s = self.G(img_s)
                feat_cr_t = self.G(img_t)

                output_cr_s_D = self.D(feat_cr_s.cuda())
                output_cr_t_D = self.D(feat_cr_t.cuda())

                loss_bce2 = nn.BCELoss()(output_cr_s_D, output_cr_t_D.detach())
                loss_2 = 0.1*loss_bce2
                loss_2.backward()
                self.opt_d.step()
                self.reset_grad()
            
            
###################### discriminablity ##################
                feat_cr_s = self.G(img_s)
                feat_cr_t = self.G(img_t)
            
                output_cr_s_C = self.C(feat_cr_s.cuda())
                output_cr_t_C = self.C(feat_cr_t.cuda())
                output_cr_s_C1 = self.C1(feat_cr_s.cuda())
                output_cr_s_C2 = self.C2(feat_cr_s.cuda()) 
                output_cr_t_C1 = self.C1(feat_cr_t.cuda())
                output_cr_t_C2 = self.C2(feat_cr_t.cuda())


                loss_cr_s = criterion(output_cr_s_C1, label_s) + criterion(output_cr_s_C2, label_s) + criterion(output_cr_s_C, label_s)
                loss_dis1_t = -self.discrepancy(output_cr_t_C1, output_cr_t_C2)
                loss_3 = loss_cr_s + loss_dis1_t 
                loss_3.backward()
                self.opt_c1.step()
                self.opt_c2.step()
                self.opt_c.step()
                self.reset_grad()
                
################ the balance of transferability and discriminability ###########
            for i3 in range(self.num_k3):
                feat_cr_s = self.G(img_s)
                feat_cr_t = self.G(img_t)
                output_cr_s_D = self.D(feat_cr_s.cuda())
                output_cr_t_D = self.D(feat_cr_t.cuda())

                loss_bce1 = -nn.BCELoss()(output_cr_s_D , output_cr_t_D.detach())
                loss_4 = 0.2*loss_bce1

                feat_cr_t = self.G(img_t)
                output_cr_t_C = self.C(feat_cr_t.cuda())
                output_cr_t_C1 = self.C1(feat_cr_t.cuda())
                output_cr_t_C2 = self.C2(feat_cr_t.cuda())

                loss_51 = self.discrepancy(output_cr_t_C1, output_cr_t_C2)
                loss_52 = self.discrepancy(output_cr_t_C, output_cr_t_C1)
                loss_53 = self.discrepancy(output_cr_t_C, output_cr_t_C2)
                loss_5 = loss_51 + loss_52 + loss_53
                
                ######## GH or GH++ ########
                if self.GH == True:
                    self.reset_grad()
                    loss_4.backward(retain_graph=True)
                    grad1 = self.getgrad(self.G)
                    self.reset_grad()
                    loss_5.backward(retain_graph=True)
                    grad2 = self.getgrad(self.G)
                    self.reset_grad()
                    if torch.dot(grad1,grad2)<0:
                        gh1=1- (torch.dot(grad1,grad2))/(torch.dot(grad1,grad1))
                        gh2=1- (torch.dot(grad1,grad2))/(torch.dot(grad2,grad2))
                    else:
                        gh1=1
                        gh2=1                  
                    total_loss = gh1 * T * loss_4 + gh2 * (1.0 - T) * loss_5
                elif self.GH_new == True:
                    self.reset_grad()
                    loss_4.backward(retain_graph=True)
                    grad1 = self.getgrad(self.G)
                    self.reset_grad()
                    loss_5.backward(retain_graph=True)
                    grad2 = self.getgrad(self.G)
                    self.reset_grad()
                    dot_product = torch.dot(grad1, grad2)
                    if dot_product<0:            
                        norm1 = torch.norm(grad1)
                        norm2 = torch.norm(grad2)
                        cosine_similarity = dot_product / (norm1 * norm2)
                        angle_radians = torch.acos(cosine_similarity)
                        angle_degrees = math.degrees(angle_radians.item())#\theta
                        #print('angle_degrees:', angle_degrees)
                        theta = angle_degrees
                        beta_max = angle_degrees - 90
                        beta = self.trade_off_1 * beta_max
                        gh1=1 + 2 * math.sin(math.radians(0.5 * beta))
                        gh2=1 + 2 * math.sin(math.radians(0.5 *(beta + 90 - theta)))
                    else:
                        gh1=1
                        gh2=1        
                    total_loss = gh1 * T * loss_4 + gh2 * (1.0 - T) * loss_5
                else:
                    total_loss = T * loss_4 + (1.0 - T) * loss_5

                total_loss.backward()
                self.opt_g.step()
                self.reset_grad()  

############ re-weiting based on the uncertainity of pseudo-label ###############
            for i4 in range(self.num_k4):
                ############# H_emp
                feat_cr_t = self.G(img_t)
                output_cr_t_C = self.C(feat_cr_t.cuda())
                output_cr_t_C_de = output_cr_t_C.detach()

                for ii in range(self.batch_size):
                    self.output_cr_t_C_label[ii] = np.argmax(output_cr_t_C_de[ii].cpu().numpy())
                output_cr_t_C_labels = torch.from_numpy(self.output_cr_t_C_label).cuda().long()
                Ly_ce_t = criterion(output_cr_t_C, output_cr_t_C_labels)
                H_emp = self.ent(output_cr_t_C)
                ############# weight coefficient mu
                mu = (torch.exp(-H_emp) - 1.0 / self.num_classes) / (1 - 1.0 / self.num_classes)
                Ly_loss = 2*(mu*Ly_ce_t+(1-mu)*H_emp)
                Ly_loss.backward()
                self.opt_g.step()
                self.opt_c.step()
                self.reset_grad()
            self.reset_grad()
     
#================ data for A_st and J_w ==============================#
            if self.data == 'Digits':
                feat_s = self.G(img_s)
                feat_t = self.G(img_t)
            else:
                feat_s = self.PoolLayer(self.G(img_s))
                feat_t = self.PoolLayer(self.G(img_t))

            #label_s
            label_predi = self.C(self.G(img_t))
            #----------------------s-----------------------#
            label_s_test = label_s
            feat_s_test = feat_s

            label_s_test_np = label_s_test.cpu().detach().numpy() 
            feat_s_test_np = feat_s_test.cpu().detach().numpy()
 
            label_s_for_LDA = np.concatenate((label_s_for_LDA,label_s_test_np),axis=0)
            fea_s_for_LDA = np.vstack((fea_s_for_LDA, feat_s_test_np))

            #-------------------------t --------------------#
            feat_test = feat_t
            feat_test_np = feat_test.cpu().detach().numpy()
            fea_for_LDA = np.vstack((fea_for_LDA,feat_test_np))
            #------------------------t_label-----------------#
            label_test = label_predi

            label_t = torch.max(label_test, 1)[1]

            label_test_np = label_t.cpu().detach().numpy()

            label_test_np = label_test_np.reshape(self.batch_size,1)
            
            label_for_LDA = np.vstack((label_for_LDA,label_test_np))


        ################## mmd ##############################
            if self.scale:
                loss_mmd = self.linear_mmd(feat_s ,feat_t )
                A_st = loss_mmd.cpu().detach().numpy()
                A_st_max = max(abs(A_st_max),abs(A_st))
                A_st_min = min(abs(A_st_min),abs(A_st))
                A_st_norm = abs(A_st-A_st_min)/(A_st_max-A_st_min+1e-6)
        if not self.scale:
            f_of_X = torch.from_numpy(fea_s_for_LDA)
            f_of_Y = torch.from_numpy(fea_for_LDA)
            loss_mmd = self.linear_mmd(f_of_X, f_of_Y)
            A_st = loss_mmd.cpu().detach().numpy()
            A_st_max = max(abs(A_st_max),abs(A_st))
            A_st_min = min(abs(A_st_min),abs(A_st))
            A_st_norm = abs(A_st-A_st_min)/(A_st_max-A_st_min+1e-6)

        ###################### J_w_s ####################################       
        n_dim = self.num_classes - 1
        clusters1 = np.unique(label_s_for_LDA)

        if n_dim > len(clusters1)-1:
            print("K is too much")
            print("please input again")
            exit(0)
        Sw1 = np.zeros((fea_s_for_LDA.shape[1],fea_s_for_LDA.shape[1]))
        for i in clusters1:
            datai1 = fea_s_for_LDA[label_s_for_LDA.reshape(-1) == i]
            datai1 = datai1 - datai1.mean(0)
            Swi1 = np.mat(datai1).T * np.mat(datai1)
            Sw1 += Swi1

	    # between_class scatter matrix
        SB1 = np.zeros((fea_s_for_LDA.shape[1],fea_s_for_LDA.shape[1]))
        u1 = fea_s_for_LDA.mean(0)  # 所有样本的平均值

        for i in clusters1:
            Ni1 = fea_s_for_LDA[label_s_for_LDA.reshape(-1) == i].shape[0]
            ui1 = fea_s_for_LDA[label_s_for_LDA.reshape(-1) == i].mean(0)  # 某个类别的平均值
            SBi1 = Ni1 * np.mat(ui1 - u1).T * np.mat(ui1 - u1)
            SB1 += SBi1
        S1 = np.linalg.inv(Sw1 + (1e-6 * np.eye(Sw1.shape[0]))) * SB1
        eigVals1,eigVects1 = np.linalg.eig(S1)  #求特征值，特征向量
        eigValInd1 = np.argsort(eigVals1)
        eigValInd1 = eigValInd1[:(-n_dim-1):-1]
        J_max1 = 0

        for i in range(n_dim):
            J_max1 = J_max1 + eigVals1[eigValInd1[i]]
        J_w_s = J_max1/self.num_classes #self.class_num
        max_J_w = max(max_J_w,J_w_s)
        min_J_w = min(min_J_w,J_w_s)
        ###################### J_w_t ####################################       
        n_dim = self.num_classes-1
        clusters = np.unique(label_for_LDA)

        if n_dim > len(clusters)-1:
            print("K is too much")
            print("please input again")
            exit(0)
            
        Sw = np.zeros((fea_for_LDA.shape[1], fea_for_LDA.shape[1]))
        for i in clusters:
            datai = fea_for_LDA[label_for_LDA.reshape(-1) == i]
            datai = datai - datai.mean(0)
            Swi = np.mat(datai).T * np.mat(datai)
            Sw += Swi

	    #between_class scatter matrix
        SB = np.zeros((fea_for_LDA.shape[1],fea_for_LDA.shape[1]))
        u = fea_for_LDA.mean(0)  #所有样本的平均值

        for i in clusters:
            Ni = fea_for_LDA[label_for_LDA.reshape(-1) == i].shape[0]
            ui = fea_for_LDA[label_for_LDA.reshape(-1) == i].mean(0)  # 某个类别的平均值
            SBi = Ni * np.mat(ui - u).T * np.mat(ui - u)
            SB += SBi

        S = np.linalg.inv(Sw+(1e-6*np.eye(Sw.shape[0])))*SB
        eigVals,eigVects = np.linalg.eig(S)  #求特征值，特征向量
        eigValInd = np.argsort(eigVals)
        eigValInd = eigValInd[:(-n_dim-1):-1]
        J_max = 0
        for i in range(n_dim):
            J_max = J_max + eigVals[eigValInd[i]]
        J_w_t = J_max/self.num_classes
        min_J_w = min(min_J_w,J_w_t)
        max_J_w = max(max_J_w,J_w_t)
        ###################### J_w_1_norm #############################
        J_w = min(J_w_s,J_w_t)
        J_w_norm = (J_w -min_J_w)/(max_J_w-min_J_w+1e-6)

        return A_st_min, A_st_max, min_J_w, max_J_w, A_st_norm, J_w_norm



#test--------------test---------------------------
    def test(self, acc_max, epoch, record_file=None, save_model=False):
        self.G.eval()
        self.C.eval()
        self.C1.eval()
        self.C2.eval()
        test_loss1 = 0
        test_loss2 = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        correct4 = 0
        correct5 = 0
        size = 0
        for batch_idx, data in enumerate(self.test_loader):
            img, label = data[:2]
            img, label = img.cuda(), label.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            feat = self.G(img)
            output1 = self.C(feat)
            output21 = self.C1(feat)
            output22 = self.C2(feat)
            test_loss1 += F.nll_loss(output1, label).item()
            test_loss2 += F.nll_loss(output21, label).item()
            output_ensemble_c1c2 = output21 + output22
            output_ensemble_cc1c2 = output1 + output21 + output22
            pred1 = output1.data.max(1)[1]
            pred21 = output21.data.max(1)[1]
            pred22 = output22.data.max(1)[1]
            pred_ensemble_c1c2 = output_ensemble_c1c2.data.max(1)[1]
            pred_ensemble_cc1c2 = output_ensemble_cc1c2.data.max(1)[1]
            k = label.data.size()[0]
            correct1 += pred1.eq(label.data).cpu().sum()
            correct2 += pred21.eq(label.data).cpu().sum()
            correct3 += pred22.eq(label.data).cpu().sum()
            correct4 += pred_ensemble_c1c2.eq(label.data).cpu().sum()
            correct5 += pred_ensemble_cc1c2.eq(label.data).cpu().sum()
            size += k
        acc1 = 100. * float(correct1) / float(size)
        acc2 = 100. * float(correct2) / float(size)
        acc3 = 100. * float(correct3) / float(size)
        acc4 = 100. * float(correct4) / float(size)
        acc5 = 100. * float(correct5) / float(size)
        acc = max(acc1,acc2,acc3,acc4,acc5)
        print(
            '*Test [Accuracy C: {}/{} ({:.1f}%)] [Max Accuracy: ({:.1f}%)]\n'.format(
                correct1, size, acc1, acc))
        if acc > acc_max:
            acc_max = acc
        #    print(
        #        'Test Epoch: {} \n Accuracy C: {}/{} ({:.1f}%), Max Accuracy: ({:.1f}%)\n'.format(
        #            epoch, correct1, size, acc1, acc))
        if save_model and epoch % self.save_epoch == 0:
            torch.save(self.G,
                       '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source[0], self.target[0], epoch))
            torch.save(self.C,
                       '%s/%s_to_%s_model_epoch%s_C.pt' % (self.checkpoint_dir, self.source[0], self.target[0], epoch))
            torch.save(self.C1,
                       '%s/%s_to_%s_model_epoch%s_C1.pt' % (self.checkpoint_dir, self.source[0], self.target[0], epoch))
            torch.save(self.C2,
                       '%s/%s_to_%s_model_epoch%s_C2.pt' % (self.checkpoint_dir, self.source[0], self.target[0], epoch))
            torch.save(self.D,
                       '%s/%s_to_%s_model_epoch%s_D.pt' % (self.checkpoint_dir, self.source[0], self.target[0], epoch))
        if record_file:
            record = open(record_file, 'a')
            record.write('%s\t%s\n' % (float(correct1) / size,  acc))
            record.close()
        return acc_max