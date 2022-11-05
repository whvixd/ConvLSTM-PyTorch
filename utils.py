import os
from collections import OrderedDict
import argparse
import torch
from torch import nn
from torchinfo import summary

parser = argparse.ArgumentParser()

# 使用：python main.py --gpu_device_ids 1,2,3
parser.add_argument('--gpu_device_ids',
                    default='0, 1, 2, 3',
                    help='gpu device ids')
args = parser.parse_args()
gpu_device_list = args.gpu_device_ids
device_ids = gpu_device_list.split(',')
# 注：使用本地服务哪几块GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device_list

cuda_available = torch.cuda.is_available()
# 模型参数存储的主卡
device = torch.device("cuda:3" if cuda_available else "cpu")


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))


def try_to_cuda(data):
    return data.cuda() if cuda_available else data


def get_device():
    return device


def print_model(model):
    summary(model)


def get_device_ids():
    return device_ids
