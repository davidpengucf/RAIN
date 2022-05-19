#!/usr/bin/env python
# coding: utf-8



import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
#import network, loss
from torch.utils.data import DataLoader
#from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix




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
parser.add_argument('--ssl', type=float, default=0.0) 
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




import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
import cv2
import torchvision





def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images



def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')



class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)



class ImageList_idx(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)



def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders




def image_train(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])




def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])




import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict




def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)



res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, 
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}





class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x




class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x




class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x



import torch
import torch.utils.data
from torchvision import datasets
import numpy as np




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



def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            #inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    mean_ent = ent.mean()
    ent_sel = ent < mean_ent
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    
    mean_ent = torch.mean(Entropy(all_output)).data.item()

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return pred_label.astype('int'), ent_sel



def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 



def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer




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
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = ResBase(res_name=args.net)#.cuda()
    elif args.net[0:3] == 'vgg':
        netF = VGGBase(vgg_name=args.net)#.cuda()  

    netB = feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck)#.cuda()
    netC = feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck)#.cuda()
    
    '''if not args.ssl == 0:
        netR = feat_classifier(type='linear', class_num=4, bottleneck_dim=2*args.bottleneck)#.cuda()
        netR_dict, acc_rot = train_target_rot(args)
        netR.load_state_dict(netR_dict)'''
    
    #modelpath = args.output_dir_src + '/source_F.pt'   
    modelpath = osp.join(args.output_dir, "target_F_par_best.pt")
    netF.load_state_dict(torch.load(modelpath))
    modelpath = osp.join(args.output_dir, "target_B_par_best.pt") 
    netB.load_state_dict(torch.load(modelpath))
    modelpath = osp.join(args.output_dir, "target_C_par_best.pt")  
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
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            mem_label, ent_sel = obtain_label(dset_loaders['test'], netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label)#.cuda()
            netF.train()
            netB.train()

        #inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        if args.cls_par > 0:
            new_idx = []
            uns_idx = []
            for ele in tar_idx:
                if ent_sel[tar_idx] == True:
                    new_idx.append(ele)
                else:
                    uns_idx.append(ele)
                    
            new_idx = torch.stack(new_idx,dim=0)
            pred = mem_label[new_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test[new_idx], pred)
            classifier_loss *= args.cls_par
            #if iter_num < interval_iter and args.dset == "VISDA-C":
                #classifier_loss *= 0
        #else:
            #classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            uns_idx = torch.stack(uns_idx,dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs_test[uns_idx])
            entropy_loss = torch.mean(Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss
            
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





