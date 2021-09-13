#  https://github.com/lshiwjx/2s-AGCN

import argparse
import inspect
import os
import pickle
import math
import yaml
import random
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=16,
        help='the number of worker for data loader')
    parser.add_argument(
        '--test-feeder-args1',
        default=dict(),
        help='the arguments of data loader for test')
    parser.add_argument(
        '--test-feeder-args2',
        default=dict(),
        help='the arguments of data loader for test')
    parser.add_argument(
        '--test-feeder-args3',
        default=dict(),
        help='the arguments of data loader for test')
    parser.add_argument(
        '--test-feeder-args4',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights1',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--weights2',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--weights3',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--weights4',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    return parser


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = get_parser()
p = parser.parse_args()
with open(p.config, 'r') as f:
    default_arg = yaml.load(f, Loader=yaml.FullLoader)

parser.set_defaults(**default_arg)
arg, unknown = parser.parse_known_args()

Feeder = import_class(arg.feeder)
test_loader1 = torch.utils.data.DataLoader(
    dataset=Feeder(**arg.test_feeder_args1), batch_size=arg.test_batch_size, shuffle=False,
    num_workers=arg.num_worker, drop_last=False, worker_init_fn=init_seed)
if arg.test_feeder_args2 is not None:
    test_loader2 = torch.utils.data.DataLoader(
        dataset=Feeder(**arg.test_feeder_args2), batch_size=arg.test_batch_size, shuffle=False,
        num_workers=arg.num_worker, drop_last=False, worker_init_fn=init_seed)
if arg.test_feeder_args3 is not None:
    test_loader3 = torch.utils.data.DataLoader(
        dataset=Feeder(**arg.test_feeder_args3), batch_size=arg.test_batch_size, shuffle=False,
        num_workers=arg.num_worker, drop_last=False, worker_init_fn=init_seed)
if arg.test_feeder_args4 is not None:
    test_loader4 = torch.utils.data.DataLoader(
        dataset=Feeder(**arg.test_feeder_args4), batch_size=arg.test_batch_size, shuffle=False,
        num_workers=arg.num_worker, drop_last=False, worker_init_fn=init_seed)

output_device = arg.device
Model = import_class(arg.model)

model1 = Model(**arg.model_args).cuda(output_device)
weights = torch.load(arg.weights1)
weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
model1.load_state_dict(weights)
model1.eval()
if arg.weights2 is not None:
    model2 = Model(**arg.model_args).cuda(output_device)
    weights = torch.load(arg.weights2)
    weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
    model2.load_state_dict(weights)
    model2.eval()
if arg.weights3 is not None:
    model3 = Model(**arg.model_args).cuda(output_device)
    weights = torch.load(arg.weights3)
    weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
    model3.load_state_dict(weights)
    model3.eval()
if arg.weights4 is not None:
    model4 = Model(**arg.model_args).cuda(output_device)
    weights = torch.load(arg.weights4)
    weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
    model4.load_state_dict(weights)
    model4.eval()

    
print("Joint model...")
acc = 0.
process = tqdm(test_loader1)
for batch_idx, (data, label, index) in enumerate(process):
    with torch.no_grad():
        label = label.long().cuda(output_device)
        data = data.float().cuda(output_device)
        
        output = nn.functional.softmax(model1(data), dim=1)
        _, predict_label = torch.max(output.data, 1)
        acc += (predict_label == label.data).sum()
acc = acc / len(test_loader1.dataset)
print("Val acc:", acc * 100.)

if arg.weights2 is not None:
    print("Bone-inward model...")
    acc = 0.
    process = tqdm(test_loader2)
    for batch_idx, (data, label, index) in enumerate(process):
        with torch.no_grad():
            label = label.long().cuda(output_device)
            data = data.float().cuda(output_device)

            output = nn.functional.softmax(model2(data), dim=1)
            _, predict_label = torch.max(output.data, 1)
            acc += (predict_label == label.data).sum()
    acc = acc / len(test_loader1.dataset)
    print("Val acc:", acc * 100.)

if arg.weights3 is not None:
    print("Bone-outward model...")
    acc = 0.
    process = tqdm(test_loader3)
    for batch_idx, (data, label, index) in enumerate(process):
        with torch.no_grad():
            label = label.long().cuda(output_device)
            data = data.float().cuda(output_device)

            output = nn.functional.softmax(model3(data), dim=1)
            _, predict_label = torch.max(output.data, 1)
            acc += (predict_label == label.data).sum()
    acc = acc / len(test_loader1.dataset)
    print("Val acc:", acc * 100.)

if arg.weights4 is not None:
    print("Bone-motion model...")
    acc = 0.
    process = tqdm(test_loader4)
    for batch_idx, (data, label, index) in enumerate(process):
        with torch.no_grad():
            label = label.long().cuda(output_device)
            data = data.float().cuda(output_device)

            output = nn.functional.softmax(model4(data), dim=1)
            _, predict_label = torch.max(output.data, 1)
            acc += (predict_label == label.data).sum()
    acc = acc / len(test_loader1.dataset)
    print("Val acc:", acc * 100.)

if arg.weights2 is not None:
    print("2s model...")
    data_iter2 = iter(test_loader2)
    acc = 0.
    process = tqdm(test_loader1)
    for batch_idx, (data1, label, index) in enumerate(process):
        data2, _, _ = next(data_iter2)

        with torch.no_grad():
            label = label.long().cuda(output_device)
            data1 = data1.float().cuda(output_device)
            data2 = data2.float().cuda(output_device)

            output1 = model1(data1)
            output2 = model2(data2)

            output = output1 + output2
            _, predict_label = torch.max(output.data, 1)
            acc += (predict_label == label.data).sum()

    acc = acc / len(test_loader1.dataset)
    print("Val acc:", acc * 100.)

if arg.weights4 is not None:
    print("4s model...")
    data_iter2 = iter(test_loader2)
    data_iter3 = iter(test_loader3)
    data_iter4 = iter(test_loader4)
    acc = 0.
    process = tqdm(test_loader1)
    for batch_idx, (data1, label, index) in enumerate(process):
        data2, _, _ = next(data_iter2)
        data3, _, _ = next(data_iter3)
        data4, _, _ = next(data_iter4)

        with torch.no_grad():
            label = label.long().cuda(output_device)
            data1 = data1.float().cuda(output_device)
            data2 = data2.float().cuda(output_device)
            data3 = data3.float().cuda(output_device)
            data4 = data4.float().cuda(output_device)

            output1 = model1(data1)
            output2 = model2(data2)
            output3 = model3(data3)
            output4 = model4(data4)

            output = output1 + output2 + output3 + output4
            _, predict_label = torch.max(output.data, 1)
            acc += (predict_label == label.data).sum()

    acc = acc / len(test_loader1.dataset)
    print("Val acc:", acc * 100.)