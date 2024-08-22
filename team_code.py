#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################
# Helper 
import torch
from torch import nn
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.transforms import functional as TF
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import re

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2        
import os
import pandas as pd
import glob
from tqdm import tqdm, trange
import subprocess
import requests

# get_file_names
def get_file_names(input_files: list) -> list:
    input_file_names = []
    for input_file in input_files:
        temp_file_name = input_file.split('/')[-1].strip('.png')
        input_file_names.append(temp_file_name)
    print ('Total', len(input_file_names), 'files')
    return input_file_names

################################################################################

import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

def compute_layer_rf_info(layer_filter_size, layer_stride, layer_padding,
                          previous_layer_rf_info):
    n_in = previous_layer_rf_info[0] # input size
    j_in = previous_layer_rf_info[1] # receptive field jump of input layer
    r_in = previous_layer_rf_info[2] # receptive field size of input layer
    start_in = previous_layer_rf_info[3] # center of receptive field of input layer

    if layer_padding == 'SAME':
        n_out = math.ceil(float(n_in) / float(layer_stride))
        if (n_in % layer_stride == 0):
            pad = max(layer_filter_size - layer_stride, 0)
        else:
            pad = max(layer_filter_size - (n_in % layer_stride), 0)
        assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
        assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check
    elif layer_padding == 'VALID':
        n_out = math.ceil(float(n_in - layer_filter_size + 1) / float(layer_stride))
        pad = 0
        assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
        assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check
    else:
        # layer_padding is an int that is the amount of padding on one side
        pad = layer_padding * 2
        n_out = math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1

    pL = math.floor(pad/2)

    j_out = j_in * layer_stride
    r_out = r_in + (layer_filter_size - 1)*j_in
    start_out = start_in + ((layer_filter_size - 1)/2 - pL)*j_in
    return [n_out, j_out, r_out, start_out]

def compute_rf_protoL_at_spatial_location(img_size, height_index, width_index, protoL_rf_info):
    n = protoL_rf_info[0]
    j = protoL_rf_info[1]
    r = protoL_rf_info[2]
    start = protoL_rf_info[3]
    assert(height_index < n)
    assert(width_index < n)

    center_h = start + (height_index*j)
    center_w = start + (width_index*j)

    rf_start_height_index = max(int(center_h - (r/2)), 0)
    rf_end_height_index = min(int(center_h + (r/2)), img_size)

    rf_start_width_index = max(int(center_w - (r/2)), 0)
    rf_end_width_index = min(int(center_w + (r/2)), img_size)

    return [rf_start_height_index, rf_end_height_index,
            rf_start_width_index, rf_end_width_index]

def compute_rf_prototype(img_size, prototype_patch_index, protoL_rf_info):
    img_index = prototype_patch_index[0]
    height_index = prototype_patch_index[1]
    width_index = prototype_patch_index[2]
    rf_indices = compute_rf_protoL_at_spatial_location(img_size,
                                                       height_index,
                                                       width_index,
                                                       protoL_rf_info)
    return [img_index, rf_indices[0], rf_indices[1],
            rf_indices[2], rf_indices[3]]

def compute_rf_prototypes(img_size, prototype_patch_indices, protoL_rf_info):
    '''
    This function tells you the receptive field (BY LOOKING AT A PARTICULAR H AND W OF FEATURE MAP)
    of prototype layer onto the actual image.
    '''
    rf_prototypes = []
    for prototype_patch_index in prototype_patch_indices:
        img_index = prototype_patch_index[0]
        height_index = prototype_patch_index[1]
        width_index = prototype_patch_index[2]
        rf_indices = compute_rf_protoL_at_spatial_location(img_size,
                                                           height_index,
                                                           width_index,
                                                           protoL_rf_info)
        rf_prototypes.append([img_index, rf_indices[0], rf_indices[1],
                              rf_indices[2], rf_indices[3]])
    return rf_prototypes

def compute_proto_layer_rf_info(img_size, cfg, prototype_kernel_size):
    rf_info = [img_size, 1, 1, 0.5]

    for v in cfg:
        if v == 'M':
            rf_info = compute_layer_rf_info(layer_filter_size=2,
                                            layer_stride=2,
                                            layer_padding='SAME',
                                            previous_layer_rf_info=rf_info)
        else:
            rf_info = compute_layer_rf_info(layer_filter_size=3,
                                            layer_stride=1,
                                            layer_padding='SAME',
                                            previous_layer_rf_info=rf_info)

    proto_layer_rf_info = compute_layer_rf_info(layer_filter_size=prototype_kernel_size,
                                                layer_stride=1,
                                                layer_padding='VALID',
                                                previous_layer_rf_info=rf_info)

    return proto_layer_rf_info

def compute_proto_layer_rf_info_v2(img_size, layer_filter_sizes, layer_strides, layer_paddings, prototype_kernel_size):

    assert(len(layer_filter_sizes) == len(layer_strides))
    assert(len(layer_filter_sizes) == len(layer_paddings))

    rf_info = [img_size, 1, 1, 0.5]

    for i in range(len(layer_filter_sizes)):
        filter_size = layer_filter_sizes[i]
        stride_size = layer_strides[i]
        padding_size = layer_paddings[i]

        rf_info = compute_layer_rf_info(layer_filter_size=filter_size,
                                layer_stride=stride_size,
                                layer_padding=padding_size,
                                previous_layer_rf_info=rf_info)

    proto_layer_rf_info = compute_layer_rf_info(layer_filter_size=prototype_kernel_size,
                                                layer_stride=1,
                                                layer_padding='VALID',
                                                previous_layer_rf_info=rf_info)

    return proto_layer_rf_info

model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

model_dir = './pretrained_models'

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_features(nn.Module):

    def __init__(self, cfg, batch_norm=False, init_weights=True):
        super(VGG_features, self).__init__()

        self.batch_norm = batch_norm

        self.kernel_sizes = []
        self.strides = []
        self.paddings = []

        self.features = self._make_layers(cfg, batch_norm)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, batch_norm):

        self.n_layers = 0

        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

                self.kernel_sizes.append(2)
                self.strides.append(2)
                self.paddings.append(0)

            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]

                self.n_layers += 1

                self.kernel_sizes.append(3)
                self.strides.append(1)
                self.paddings.append(1)

                in_channels = v

        return nn.Sequential(*layers)

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def num_layers(self):
        '''
        the number of conv layers in the network
        '''
        return self.n_layers

    def __repr__(self):
        template = 'VGG{}, batch_norm={}'
        return template.format(self.num_layers() + 3,
                               self.batch_norm)

def vgg19_features(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['E'], batch_norm=False, **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['vgg19'], model_dir=model_dir)
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    return model

base_architecture_to_features = {                            'vgg19': vgg19_features}

class PPNet(nn.Module):

    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4
        
        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        # (used for setting weights of last layer)
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias

        # Sigmoid activation to output probabilities
        self.sigmoid = nn.Sigmoid()

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x = self.add_on_layers(x)
        return x

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)
        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x):
        distances = self.prototype_distances(x)
        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''
        # global min pooling
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = self.last_layer(prototype_activations)
        probs = self.sigmoid(logits)
        return probs, min_distances

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output = self.conv_features(x)
        distances = self._l2_convolution(conv_output)
        return conv_output, distances

    def prune_prototypes(self, prototypes_to_prune):
        '''
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...],
                                              requires_grad=True)

        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...],
                                 requires_grad=False)
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)



def construct_PPNet(base_architecture='vgg19', pretrained=False, img_size=224,
                    prototype_shape=(352, 128, 1, 1), num_classes=11,
                    prototype_activation_function='log',
                    add_on_layers_type='regular'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    return PPNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)

################################################################################



# Conv backbone
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

#UNet model
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)
        self.down3 = nn.MaxPool2d(2)
        self.down4 = nn.MaxPool2d(2)
        self.double_conv1 = DoubleConv(64, 128)
        self.double_conv2 = DoubleConv(128, 256)
        self.double_conv3 = DoubleConv(256, 512)
        self.double_conv4 = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.double_conv_up1 = DoubleConv(1024, 512)
        self.double_conv_up2 = DoubleConv(512, 256)
        self.double_conv_up3 = DoubleConv(256, 128)
        self.double_conv_up4 = DoubleConv(128, 64)
        self.sigmoid = nn.Sigmoid()
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.double_conv1(x2)
        x3 = self.down2(x2)
        x3 = self.double_conv2(x3)
        x4 = self.down3(x3)
        x4 = self.double_conv3(x4)
        x5 = self.down4(x4)
        x5 = self.double_conv4(x5)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.double_conv_up1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.double_conv_up2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.double_conv_up3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.double_conv_up4(x)
        logits = self.outc(x)
        logits = self.sigmoid(logits)
        return logits

#Dice loss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

#Focal Loss https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Focal-Loss
class FocalLoss(nn.Module): 
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha, gamma)

    def forward(self, inputs, targets):
        return self.dice_loss(inputs, targets) + self.focal_loss(inputs, targets)


def calculate_dice(target, prediction):
    """
    Calculate Dice Coefficient for a single image.
    
    Args:
        target (np.array): The ground truth binary segmentation image.
        prediction (np.array): The predicted binary segmentation image.
    
    Returns:
        float: The Dice Coefficient.
    """
    intersection = np.logical_and(target, prediction)
    dice_score = 2 * np.sum(intersection) / (np.sum(target) + np.sum(prediction))
    return dice_score

def save_model_layers(model, directory="model_layers"):
    os.makedirs(directory, exist_ok=True)
    for name, param in model.named_parameters():
        file_path = os.path.join(directory, f"{name.replace('.', '_')}.pt")
        torch.save(param.data, file_path)

def load_model_layers(model, directory="model_layers"):
    for name, param in model.named_parameters():
        file_path = os.path.join(directory, f"{name.replace('.', '_')}.pt")
        if os.path.exists(file_path):
            param_data = torch.load(file_path, map_location=torch.device('cpu'))
            param.data.copy_(param_data)

# def get_classification_model():
#     model = models.resnet18(weights=None)
#     model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     model.conv1.weight.data = model.conv1.weight.data.mean(dim=1, keepdim=True)

#     # Modify the fully connected layer for binary classification
#     num_features = model.fc.in_features
#     model.fc = torch.nn.Linear(num_features, 11)  # Adjusting for binary classification
#     load_model_layers(model, directory='/challenge/pretrain_model/ResNet18/')
#     return model


def get_classification_model():
    model = construct_PPNet()
    load_model_layers(model, directory='/challenge/pretrain_model/Ppnet/')
    return model

def get_denoise_model():
    model = UNet(n_channels=3, n_classes=1)
    load_model_layers(model, directory='/challenge/pretrain_model/UNet/')
    return model

# End of Helper

# Weijie_model

# device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # Adjusted for a single channel
])

noise_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization values
])

class ECGNoisedImageDataset(Dataset):
    def __init__(self, image_path_list, transform = None, 
                 noise_transform = None, denoise_model = None, device = None):
        """
        Args:
            dataframe (pd.DataFrame): A pandas dataframe containing 'file_path' and 'label' columns.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_path_list = image_path_list
        # self.label_list = label_list
        self.noise_transform = noise_transform
        self.transform = transform
        self.denoise_model = denoise_model
        self.device = device

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):

        
        input_image = cv2.imread(self.image_path_list[idx], cv2.IMREAD_COLOR)
        input_image = self.noise_transform(input_image)        
        input_image = TF.resize(input_image, (896, 1152))
        input_image = torch.sigmoid(input_image).to(device)
        input_image = input_image.reshape(1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
        output_image = self.denoise_model(input_image)

        if self.transform:
            output_image = self.transform(output_image)

        # label = self.label_list[idx]
        return output_image #, torch.tensor([label, ~label]).float(), self.image_path_list[idx]

def get_current_directory():
    return os.path.dirname(os.path.abspath(__file__))

def load_data_weijie(data_path):
    
    current_directory = get_current_directory()    
    # PATH = '/home/weijiesun/physinet2024/model/UNet_torch.model' #./model/UNet_torch.model'
    # Weijie Need to update
    denoise_model = get_denoise_model().to(device)
    # denoise_model = torch.load(path_unet, map_location ='cpu').to(device)
    # denoise_model.eval()
    
    # PATH = '/home/weijiesun/physinet2024/model/ResNet_C_torch.model' #'./model/ResNet_C_torch.model'
    model = get_classification_model().to(device)
    # model.eval()

    # pattern = re.compile(r'.*\.(png|jpg|jpeg)$')
    # temp_list = [f for f in os.listdir(data_path) if pattern.match(f)]
    # temp_path = data_path + '-0.png'
    temp_list = [data_path]
    # print(data_path)
    # temp_list = [temp_list[0]]
    # print ('Weijie: data_path',data_path)
    # print(f'temp_list - new: datapath: {data_path} -- temp_list: {temp_list}')
    # print ('Weijie: feel there is a problem', len(temp_list), data_path)

    batch_size = 1
    # temp_list = temp_list[:1]
    test_dataset = ECGNoisedImageDataset(image_path_list=temp_list, transform = transformations, 
                                         noise_transform = noise_transform, denoise_model = denoise_model, device = device)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # thresholds = [0.36225042,
    #  0.04905554,
    #  0.29706085,
    #  0.24501804,
    #  0.3530571,
    #  0.12362037,
    #  0.029822709,
    #  0.10859017,
    #  0.08684414,
    #  0.0737353,
    #  0.070395336]

    thresholds = [0.5497604, 0.5046783, 0.65005696, 0.5157597, 0.64057297, 0.5241727, 0.5075525, 0.50942767, 0.5161609, 0.5290637, 0.5037822]
    
    # {'NORM': 0.5497604,
    #  'Acute MI': 0.5046783,
    #  'Old MI': 0.65005696,
    #  'STTC': 0.5157597,
    #  'CD': 0.64057297,
    #  'HYP': 0.5241727,
    #  'PAC': 0.5075525,
    #  'PVC': 0.50942767,
    #  'AFIB/AFL': 0.5161609,
    #  'TACHY': 0.5290637,
    #  'BRADY': 0.5037822}
    
    labels = ['NORM', 'Acute MI', 'Old MI', 'STTC', 'CD', 'HYP', 'PAC', 'PVC', 'AFIB/AFL', 'TACHY', 'BRADY']
    
    prob_list = []
    target_list = []
    pred_binary_list = []
    with tqdm(total=len(test_dataloader)) as pbar:
        for inputs in test_dataloader:
            inputs = inputs.to(device)
            inputs = inputs.squeeze(2)
            inputs = inputs.repeat(1, 3, 1, 1)                   
            # inputs = inputs.reshape(1, 1, 224, 224)
            outputs = model(inputs)[0]
            # print(outputs)
            # _, pred_binary = torch.max(outputs, 1)
            # probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
            # targets = targets[:, 1]
            outputs = torch.sigmoid(outputs)
            pred_prob = outputs.detach().cpu().numpy()
            pred_binary = pred_prob > thresholds
            pred_binary = pred_binary.astype(int)
            
            prob_list.extend(pred_prob)
            # target_list.extend(targets.detach().cpu().numpy())
            pred_binary_list.extend(pred_binary)
            pbar.update(1)
    # print(f'pred_binary_list: {pred_binary_list}')
    # modified_list = ['Abnormal' if x == 1 else 'Normal' for x in pred_binary_list]
    # print(f'pred_binary_list: {modified_list}')
    modified_list = []

    for i in range(len(pred_binary_list[0])):
        if pred_binary_list[0][i]:
            modified_list.append(labels[i])

    return modified_list

# Train your digitization and classification models.
def train_models(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    # print(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Train the digitization model. If you are not training a digitization model, then you can remove this part of the code.

    if verbose:
        print('Training the digitization model...')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    # digitization_features = list()
    # classification_features = list()
    # classification_labels = list()

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        # record = os.path.join(data_folder, records[i])

        # Extract the features from the image; this simple example uses the same features for the digitization and classification
        # tasks.
        # features = extract_features(record)
        
        # digitization_features.append(features)

        # Some images may not be labeled...
        # labels = load_labels(record)
        # if any(label for label in labels):
            # classification_features.append(features)
            # classification_labels.append(labels)

    # ... but we expect some images to be labeled for classification.
    # if not classification_labels:
        # raise Exception('There are no labels for the data.')

    # Train the models.
    # if verbose:
        # print('Training the models on the data...')

    # Train the digitization model. This very simple model uses the mean of these very simple features as a seed for a random number
    # generator.
    # digitization_model = np.mean(features)

    # Train the classification model. If you are not training a classification model, then you can remove this part of the code.
    
    # This very simple model trains a random forest model with these very simple features.
    # classification_features = np.vstack(classification_features)
    # classes = sorted(set.union(*map(set, classification_labels)))
    # classification_labels = compute_one_hot_encoding(classification_labels, classes)

    # Define parameters for random forest classifier and regressor.
    # n_estimators   = 12  # Number of trees in the forest.
    # max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    # random_state   = 56  # Random state; set for reproducibility.

    # Fit the model.
    # classification_model = RandomForestClassifier(
    #    n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(classification_features, classification_labels)

    # Create a folder for the models if it does not already exist.
    # os.makedirs(model_folder, exist_ok=True)

    digitization_filename = os.path.join("/challenge/pretrain_model/", 'digitization_model.sav')
    # print(digitization_filename)
    digitization_model = joblib.load(digitization_filename)
    d = digitization_model
    filename = os.path.join(model_folder, 'digitization_model.sav')
    # print(filename)
    joblib.dump(d, filename, protocol=0)
    
    # Save the models.
    # save_models(model_folder, digitization_model, classification_model, classes)

    if verbose:
        # print ("Weijie - We should not train the data, we use the pretrain data")
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_models(model_folder, verbose):
    digitization_filename = os.path.join(model_folder, 'digitization_model.sav')
    digitization_model = joblib.load(digitization_filename)
    # print(device)
    classification_model = get_classification_model().to(device)
    classification_model.eval()
    return digitization_model, classification_model

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you did not train one of the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):
    # Run the digitization model; if you did not train this model, then you can set signal = None.

    # Load the digitization model.
    model = digitization_model['model']
    # print(model)

    # Load the dimensions of the signal.
    header_file = get_header_file(record)
    header = load_text(header_file)

    num_samples = get_num_samples(header)
    num_signals = get_num_signals(header)

    # print(record)
    # Extract the features.
    # try:
    #     features = extract_features(record + "-0.png")
    # except RuntimeError as e:
    #     features = extract_features(record)
    # features = features.reshape(1, -1)

    # For a overly simply minimal working example, generate "random" waveforms.
    seed = int(round(model))
    signal = np.random.default_rng(seed=seed).uniform(low=-10, high=10, size=(num_samples, num_signals))
    signal = np.asarray(signal, dtype=np.float32)
    
    # Run the classification model; if you did not train this model, then you can set labels = None.
    # Split the path by '/'
    # parts = path.split('/')
    
    # Remove the last portion
    # new_path = '/'.join(parts[:-1])
    # print(f'run_classification_model -> data_record: {record}')
    # print(f'record: {record} -- new_path: {new_path}')

    image_file_name = get_image_files(record)
    parts = record.split('/')
    new_path = '/'.join(parts[:-1])
    #print(image_file_name)
    #print(record)
    #print(new_path + '/' + image_file_name[0])
    #label = load_labels(record)
    #print(label)
    try:
        labels = load_data_weijie(new_path + '/' + image_file_name[0])
    except:
        labels = load_data_weijie(record)
    # print(labels)
    return signal, labels

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract features.
def extract_features(record):
    print ('Weijie - Should not use this extract_features function')
    images = load_images(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])

# Save your trained models.
def save_models(model_folder, digitization_model=None, classification_model=None, classes=None):
    print ('Weijie - Should not save the train models')
    if digitization_model is not None:
        d = {'model': digitization_model}
        filename = os.path.join(model_folder, 'digitization_model.sav')
        joblib.dump(d, filename, protocol=0)

    if classification_model is not None:
        d = {'model': classification_model, 'classes': classes}
        filename = os.path.join(model_folder, 'classification_model.sav')
        joblib.dump(d, filename, protocol=0)
