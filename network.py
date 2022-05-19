import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
import time



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




model_profiling_hooks = []
model_profiling_speed_hooks = []

name_space = 95
params_space = 15
macs_space = 15
seconds_space = 15

num_forwards = 10



class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.time = self.end - self.start
        if self.verbose:
            print('Elapsed time: %f ms.' % self.time)



def get_params(self):
    """get number of params in module"""
    return np.sum(
        [np.prod(list(w.size())) for w in self.parameters()])


def run_forward(self, input):
    with Timer() as t:
        for _ in range(num_forwards):
            self.forward(*input)
            #torch.cuda.synchronize()
    return int(t.time * 1e9 / num_forwards)



def conv_module_name_filter(name):
    """filter module name to have a short view"""
    filters = {
        'kernel_size': 'k',
        'stride': 's',
        'padding': 'pad',
        'bias': 'b',
        'groups': 'g',
    }
    for k in filters:
        name = name.replace(k, filters[k])
    return name


def module_profiling(self, input, output, verbose):
    ins = input[0].size()
    outs = output.size()
    # NOTE: There are some difference between type and isinstance, thus please
    # be careful.
    t = type(self)
    if isinstance(self, nn.Conv2d):
        self.n_macs = (ins[1] * outs[1] *
                       self.kernel_size[0] * self.kernel_size[1] *
                       outs[2] * outs[3] // self.groups) * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = run_forward(self, input)
        self.name = conv_module_name_filter(self.__repr__())
    elif isinstance(self, nn.ConvTranspose2d):
        self.n_macs = (ins[1] * outs[1] *
                       self.kernel_size[0] * self.kernel_size[1] *
                       outs[2] * outs[3] // self.groups) * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = run_forward(self, input)
        self.name = conv_module_name_filter(self.__repr__())
    elif isinstance(self, nn.Linear):
        self.n_macs = ins[1] * outs[1] * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = run_forward(self, input)
        self.name = self.__repr__()
    elif isinstance(self, nn.AvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_seconds = run_forward(self, input)
        self.name = self.__repr__()
    elif isinstance(self, nn.AdaptiveAvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_seconds = run_forward(self, input)
        self.name = self.__repr__()
    else:
        # This works only in depth-first travel of modules.
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        num_children = 0
        for m in self.children():
            self.n_macs += getattr(m, 'n_macs', 0)
            self.n_params += getattr(m, 'n_params', 0)
            self.n_seconds += getattr(m, 'n_seconds', 0)
            num_children += 1
        ignore_zeros_t = [
            nn.BatchNorm2d, nn.Dropout2d, nn.Dropout, nn.Sequential,
            nn.ReLU6, nn.ReLU, nn.MaxPool2d,
            nn.modules.padding.ZeroPad2d, nn.modules.activation.Sigmoid,
        ]
        if (not getattr(self, 'ignore_model_profiling', False) and
                self.n_macs == 0 and
                t not in ignore_zeros_t):
            print(
                'WARNING: leaf module {} has zero n_macs.'.format(type(self)))
        return
    if verbose:
        print(
            self.name.ljust(name_space, ' ') +
            '{:,}'.format(self.n_params).rjust(params_space, ' ') +
            '{:,}'.format(self.n_macs).rjust(macs_space, ' ') +
            '{:,}'.format(self.n_seconds).rjust(seconds_space, ' '))
    return


def add_profiling_hooks(m, verbose):
    global model_profiling_hooks
    model_profiling_hooks.append(
        m.register_forward_hook(lambda m, input, output: module_profiling(
            m, input, output, verbose=verbose)))


def remove_profiling_hooks():
    global model_profiling_hooks
    for h in model_profiling_hooks:
        h.remove()
    model_profiling_hooks = []


def model_profiling(model, height, width, batch=1, channel=3, use_cuda=False,
                    verbose=True):
    """ Pytorch model profiling with input image size
    (batch, channel, height, width).
    The function exams the number of multiply-accumulates (n_macs).

    Args:
        model: pytorch model
        height: int
        width: int
        batch: int
        channel: int
        use_cuda: bool

    Returns:
        macs: int
        params: int

    """
    model.eval()
    data = torch.rand(batch, channel, height, width)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    data = data.to(device)
    model.apply(lambda m: add_profiling_hooks(m, verbose=verbose))
    print(
        'Item'.ljust(name_space, ' ') +
        'params'.rjust(macs_space, ' ') +
        'macs'.rjust(macs_space, ' ') +
        'nanosecs'.rjust(seconds_space, ' '))
    if verbose:
        print(''.center(
            name_space + params_space + macs_space + seconds_space, '-'))
    model(data)
    if verbose:
        print(''.center(
            name_space + params_space + macs_space + seconds_space, '-'))
    print(
        'Total'.ljust(name_space, ' ') +
        '{:,}'.format(model.n_params).rjust(params_space, ' ') +
        '{:,}'.format(model.n_macs).rjust(macs_space, ' ') +
        '{:,}'.format(model.n_seconds).rjust(seconds_space, ' '))
    remove_profiling_hooks()
    return model.n_macs, model.n_params



def profiling(model, use_cuda=False):
    """profiling on either gpu or cpu"""
    print('Start model profiling, use_cuda:{}.'.format(use_cuda))
    for width_mult in sorted([0.9, 1.0], reverse=True):
        model.apply(
            lambda m: setattr(m, 'width_mult', width_mult))
        print('Model profiling with width mult {}x:'.format(width_mult))
        verbose = width_mult == max([0.9, 1.0])
        model_profiling(
            model, 224, 224,
            verbose=True)

        
def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v





class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True,
                 us=[True, True], ratio=[1, 1]):
        in_channels_max = in_channels
        out_channels_max = out_channels
        if us[0]:
            in_channels_max = int(make_divisible(
                in_channels
                * width_mult
                / ratio[0]) * ratio[0])
        if us[1]:
            out_channels_max = int(make_divisible(
                out_channels
                * width_mult
                / ratio[1]) * ratio[1])
        groups = in_channels_max if depthwise else 1
        super(USConv2d, self).__init__(
            in_channels_max, out_channels_max,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_basic = in_channels
        self.out_channels_basic = out_channels
        self.width_mult = None
        self.us = us
        self.ratio = ratio

    def forward(self, input):
        in_channels = self.in_channels_basic
        out_channels = self.out_channels_basic
        if self.us[0]:
            in_channels = int(make_divisible(
                self.in_channels_basic
                * self.width_mult
                / self.ratio[0]) * self.ratio[0])
        if self.us[1]:
            out_channels = int(make_divisible(
                self.out_channels_basic
                * self.width_mult
                / self.ratio[1]) * self.ratio[1])
        self.groups = in_channels if self.depthwise else 1
        weight = self.weight[:out_channels, :in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        if getattr(FLAGS, 'conv_averaged', False):
            y = y * (max(self.in_channels_list)/self.in_channels)
        return y



class USLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, us=[True, True]):
        in_features_max = in_features
        out_features_max = out_features
        if us[0]:
            in_features_max = make_divisible(
                in_features * width_mult)
        if us[1]:
            out_features_max = make_divisible(
                out_features * width_mult)
        super(USLinear, self).__init__(
            in_features_max, out_features_max, bias=bias)
        self.in_features_basic = in_features
        self.out_features_basic = out_features
        self.width_mult = None
        self.us = us

    def forward(self, input):
        in_features = self.in_features_basic
        out_features = self.out_features_basic
        if self.us[0]:
            in_features = make_divisible(
                self.in_features_basic * self.width_mult)
        if self.us[1]:
            out_features = make_divisible(
                self.out_features_basic * self.width_mult)
        weight = self.weight[:out_features, :in_features]
        if self.bias is not None:
            bias = self.bias[:out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, ratio=1):
        num_features_max = int(make_divisible(
            num_features * width_mult / ratio) * ratio)
        super(USBatchNorm2d, self).__init__(
            num_features_max, affine=True, track_running_stats=False)
        self.num_features_basic = num_features
        # for tracking log during training
        self.bn = nn.ModuleList(
            [nn.BatchNorm2d(i, affine=False)
             for i in [
                     int(make_divisible(
                         num_features * width_mult / ratio) * ratio)
                     for width_mult in FLAGS.width_mult_list]
             ]
        )
        self.ratio = ratio
        self.width_mult = None
        self.ignore_model_profiling = True

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        c = int(make_divisible(
            self.num_features_basic * self.width_mult / self.ratio) * self.ratio)
        if self.width_mult in FLAGS.width_mult_list:
            idx = FLAGS.width_mult_list.index(self.width_mult)
            y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean[:c],
                self.bn[idx].running_var[:c],
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        else:
            y = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        return y




def adjust_bn_layers(module):
      
    if isinstance(module, nn.BatchNorm2d):
        
        # Removing the stats computed using exponential running average
        # and resetting count
        module.reset_running_stats()
        
        # Doing this so that we can restore it later
        module._old_momentum = module.momentum
        
        # Switching to cumulutive running average
        module.momentum = 0.1
        
        # This is necessary -- because otherwise the
        # newly observed batches will not be considered
        module._old_training = module.training
        module._old_track_running_stats = module.track_running_stats
        
        module.training = True
        # module.affine = True
        module.track_running_stats = True
            


def restore_original_settings_of_bn_layers(module):
      
    if isinstance(module, nn.BatchNorm2d):
        
        # Restoring old settings
        module.momentum = module._old_momentum
        module.training = module._old_training
        module.track_running_stats = module._old_track_running_stats

def adjust_momentum(module, t):
    if isinstance(module, nn.BatchNorm2d):
        module.momentum = 1 / (t+1)



def ComputeBN(net, postloader, resolution, num_batch=8):
    net.train()
    net.apply(adjust_bn_layers)
    with torch.no_grad():
        for batch_idx, (inputs, targets, idx) in enumerate(postloader):
            img = inputs#.cuda()
            net.apply(lambda m: adjust_momentum(m, batch_idx))
            _ = net(F.interpolate(img, (resolution, resolution), mode='bilinear', align_corners=True))
            if not batch_idx < num_batch:
                break
    net.apply(restore_original_settings_of_bn_layers)
    net.eval()
    return net
