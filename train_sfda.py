import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix

from data_prepare import *
from loss import *
from FFT import * 
from network import *

def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)

def tensor_rot_180(x):
    return x.flip(2).flip(1)

def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)


def rotate_single_with_label(img, label):
    if label == 1:
        img = tensor_rot_90(img)
    elif label == 2:
        img = tensor_rot_180(img)
    elif label == 3:
        img = tensor_rot_270(img)
    
    return img
  
def rotate_batch_with_labels(batch, labels):
    images = []
    for img, label in zip(batch, labels):
        img = rotate_single_with_label(img, label)
        images.append(img.unsqueeze(0))
    
    return torch.cat(images)
  
def cal_acc_rot(loader, netF, netB, netR):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]#.cuda()
            r_labels = np.random.randint(0, 4, len(inputs))
            r_inputs = rotate_batch_with_labels(inputs, r_labels)
            r_labels = torch.from_numpy(r_labels)
            #r_inputs = r_inputs.cuda()
            
            f_outputs = netB(netF(inputs))
            f_r_outputs = netB(netF(r_inputs))

            r_outputs = netR(torch.cat((f_outputs, f_r_outputs), 1))
            if start_test:
                all_output = r_outputs.float()#.cpu()
                all_label = r_labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, r_outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, r_labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    return accuracy*100



def train_target_rot(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = ResBase(res_name=args.net)#.cuda()
    elif args.net[0:3] == 'vgg':
        netF = VGGBase(vgg_name=args.net)#.cuda()  

    netB = feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck)#.cuda()
    netR = feat_classifier(type='linear', class_num=4, bottleneck_dim=2*args.bottleneck)#.cuda()

    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    netF.eval()
    for k, v in netF.named_parameters():
        v.requires_grad = False
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    netB.eval()
    for k, v in netB.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netR.named_parameters():
        param_group += [{'params': v, 'lr': args.lr*1}]
    netR.train()
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // 10
    iter_num = 0

    rot_acc = 0
    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        #inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        r_labels_target = np.random.randint(0, 4, len(inputs_test))
        r_inputs_target = rotate_batch_with_labels(inputs_test, r_labels_target)
        r_labels_target = torch.from_numpy(r_labels_target)#.cuda()
        #r_inputs_target = r_inputs_target.cuda()
        
        f_outputs = netB(netF(inputs_test))
        f_r_outputs = netB(netF(r_inputs_target))
        r_outputs_target = netR(torch.cat((f_outputs, f_r_outputs), 1))

        rotation_loss = nn.CrossEntropyLoss()(r_outputs_target, r_labels_target)
        rotation_loss.backward() 

        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netR.eval()
            acc_rot = cal_acc_rot(dset_loaders['target'], netF, netB, netR)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_rot)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netR.train()

            if rot_acc < acc_rot:
                rot_acc = acc_rot
                best_netR = netR.state_dict()

    log_str = 'Best Accuracy = {:.2f}%'.format(rot_acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return best_netR, rot_acc
  
  
  def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            #inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float()#.cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(all_output)).data.item()
   
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent



def train_target(args):
    dset_loaders = data_load_list(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = ResBase(res_name=args.net)#.cuda()
    elif args.net[0:3] == 'vgg':
        netF = VGGBase(vgg_name=args.net)#.cuda()  

    netB = feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck)#.cuda()
    netC = feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck)#.cuda()
    
    if not args.ssl == 0:
        netR = feat_classifier(type='linear', class_num=4, bottleneck_dim=2*args.bottleneck)#.cuda()
        netR_dict, acc_rot = train_target_rot(args)
        netR.load_state_dict(netR_dict)
    
    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:
        
        optimizer.zero_grad()
        
        try:
            inputs_test_list, _, tar_idx = iter_test.next()
            inputs_test = inputs_test_list[0]
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test_list, _, tar_idx = iter_test.next()
            inputs_test = inputs_test_list[0]

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label)#.cuda()
            netF.train()
            netB.train()

        #inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss
            
        index = torch.randperm(inputs_test.shape[0])
        inputs_asst = inputs_test[index,:,:,:]
        
        #amp_asst, _ = divide_spectrum(inputs_asst)
        _, pha_asst = divide_spectrum(inputs_asst)
        inputs_pseudo = pha_space_interpolation(inputs_test, pha_asst, L=1 , ratio=0)
        features_pseudo = netB(netF(inputs_pseudo))
        outputs_pseudo = netC(features_pseudo)
        
        l1_loss = torch.mean(torch.abs(F.softmax(outputs_test, dim=1) - F.softmax(outputs_pseudo, dim=1)) )
        
        classifier_loss += l1_loss
            
        classifier_loss.backward()
            
        if not args.ssl == 0:
            r_labels_target = np.random.randint(0, 4, len(inputs_test))
            r_inputs_target = rotate_batch_with_labels(inputs_test, r_labels_target)
            r_labels_target = torch.from_numpy(r_labels_target)#.cuda()
            #r_inputs_target = r_inputs_target.cuda()

            f_outputs = netB(netF(inputs_test))
            f_outputs = f_outputs.detach()
            f_r_outputs = netB(netF(r_inputs_target))
            r_outputs_target = netR(torch.cat((f_outputs, f_r_outputs), 1))

            rotation_loss = args.ssl * nn.CrossEntropyLoss()(r_outputs_target, r_labels_target)   
            rotation_loss.backward() 
            
        max_width = args.width_mult_range[1] #### to do
        netF.apply(lambda m: setattr(m, 'width_mult', max_width))
        max_output = netC(netB(netF(inputs_test)))
        max_output_detach = max_output.detach()
        # do other widths and resolution
        min_width = args.width_mult_range[0]
        width_mult_list = [min_width]
        sampled_width = list(np.random.uniform(args.width_mult_range[0], args.width_mult_range[1], 2))
        width_mult_list.extend(sampled_width)
        for width_mult in sorted(width_mult_list, reverse=True):
            netF.apply(
                lambda m: setattr(m, 'width_mult', width_mult))
            output = netC(netB(netF(inputs_test_list[random.randint(0, 3)])))
            kd_loss = args.ssl * torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output, dim=1), F.softmax(max_output_detach, dim=1)) 
            kd_loss.backward()

        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netF.train()
            netB.train()

    if args.issave:   
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        
    return netF, netB, netC


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='SHOT')
  parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
  parser.add_argument('--s', type=int, default=0, help="source")
  parser.add_argument('--t', type=int, default=1, help="target")
  parser.add_argument('--max_epoch', type=int, default=30, help="max iterations")
  parser.add_argument('--interval', type=int, default=120)
  parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
  parser.add_argument('--worker', type=int, default=4, help="number of workers")
  parser.add_argument('--dset', type=str, default='office', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
  parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
  parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
  parser.add_argument('--seed', type=int, default=2020, help="random seed")

  parser.add_argument('--gent', type=bool, default=True)
  parser.add_argument('--ent', type=bool, default=True)
  parser.add_argument('--threshold', type=int, default=0)
  parser.add_argument('--cls_par', type=float, default=0.3)
  parser.add_argument('--ent_par', type=float, default=1.0)
  parser.add_argument('--lr_decay1', type=float, default=0.1)
  parser.add_argument('--lr_decay2', type=float, default=1.0)

  parser.add_argument('--bottleneck', type=int, default=256)
  parser.add_argument('--epsilon', type=float, default=1e-5)
  parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
  parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
  parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
  parser.add_argument('--output', type=str, default='san')
  parser.add_argument('--output_src', type=str, default='san')
  parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
  parser.add_argument('--ssl', type=float, default=0.6) 
  parser.add_argument('--issave', type=bool, default=True)
  args = parser.parse_args(args=[])

  if args.dset == 'office-home':
      names = ['Art', 'Clipart', 'Product', 'Real_World']
      args.class_num = 65 
  if args.dset == 'office':
      names = ['amazon', 'dslr', 'webcam']
      args.class_num = 31
  if args.dset == 'VISDA-C':
      names = ['train', 'validation']
      args.class_num = 12
  if args.dset == 'office-caltech':
      names = ['amazon', 'caltech', 'dslr', 'webcam']
      args.class_num = 10


 


  args.width_mult_range = [0.9, 1.0]
  args.width_mult_list = [0.9, 1.0]




  #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  SEED = args.seed
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)




  def print_args(args):
      s = "==========================================\n"
      for arg, content in args.__dict__.items():
          s += "{}:{}\n".format(arg, content)
      return s



  print('Called with args:')
  print(args)
  
  args.t = 1

  folder = './data/'
  args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
  args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
  args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

  if args.dset == 'office-home':
      if args.da == 'pda':
          args.class_num = 65
          args.src_classes = [i for i in range(65)]
          args.tar_classes = [i for i in range(25)]

  args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
  args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
  args.name = names[args.s][0].upper()+names[args.t][0].upper()

  if not osp.exists(args.output_dir):
      os.system('mkdir -p ' + args.output_dir)
  if not osp.exists(args.output_dir):
      os.mkdir(args.output_dir)

  args.savename = 'par_' + str(args.cls_par)
  if args.da == 'pda':
      args.gent = ''
      args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
  args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
  args.out_file.write(print_args(args)+'\n')
  args.out_file.flush()
  train_target(args)
