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

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/train_joint.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=16,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
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
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--subdivision', type=int, default=1, help='training subdivision')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
class Poly_LR_Scheduler(object):
    def __init__(self, base_lr, num_epochs, iters_per_epoch=0, warmup_epochs=0):
        self.lr = base_lr
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch):
        T = epoch * self.iters_per_epoch + i
        lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)

        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('Epoches %i, learning rate = %.4f' % (epoch+1, lr))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * 10
            
class Cosine_Anneal_Scheduler(object):
    def __init__(self, max_lr, min_lr, period, iters_per_epoch=0, warmup_epochs=0, gamma=0.1):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.iters_per_epoch = iters_per_epoch
        self.N = period * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.gamma = gamma

    def __call__(self, optimizer, i, epoch):
        T = epoch * self.iters_per_epoch + i
        max_lr = self.max_lr * (self.gamma ** (T // self.N))
        lr = self.min_lr + 0.5 * (max_lr - self.min_lr) * (1 + math.cos(math.pi * T / self.N))

        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = self.max_lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('Epoches %i, learning rate = %.4f' % (epoch+1, lr))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * 10


parser = get_parser()
p = parser.parse_args()
with open(p.config, 'r') as f:
    default_arg = yaml.load(f, Loader=yaml.FullLoader)

parser.set_defaults(**default_arg)
arg, unknown = parser.parse_known_args()


Feeder = import_class(arg.feeder)
batch_size = arg.batch_size // arg.subdivision
train_loader = torch.utils.data.DataLoader(
    dataset=Feeder(**arg.train_feeder_args), batch_size=batch_size, shuffle=True,
    num_workers=arg.num_worker, drop_last=True, worker_init_fn=init_seed)
test_loader = torch.utils.data.DataLoader(
    dataset=Feeder(**arg.test_feeder_args), batch_size=arg.test_batch_size, shuffle=False,
    num_workers=arg.num_worker, drop_last=False, worker_init_fn=init_seed)


output_device = arg.device
Model = import_class(arg.model)
model = Model(**arg.model_args).cuda(output_device)
criterion = nn.CrossEntropyLoss().cuda(output_device)
optimizer = optim.SGD(model.parameters(), lr=arg.base_lr, momentum=0.9,
    nesterov=arg.nesterov, weight_decay=arg.weight_decay)

def adjust_learning_rate(optimizer, epoch):
    if epoch < arg.warm_up_epoch:
        lr = arg.base_lr * (epoch + 1) / arg.warm_up_epoch
    else:
        lr = arg.base_lr * (0.1 ** np.sum(epoch >= np.array(arg.step)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


best_acc = 0.
subdivision = arg.subdivision

for epoch in range(arg.start_epoch, arg.num_epoch):
    print("Epoch: {:>2d}".format(epoch + 1), flush=True)
    model.train()
    loss_value = 0.
    if arg.only_train_part:
        if epoch > arg.only_train_epoch:
            for key, value in model.named_parameters():
                if 'PA' in key:
                    value.requires_grad = True
        else:
            for key, value in model.named_parameters():
                if 'PA' in key:
                    value.requires_grad = False
                    
    adjust_learning_rate(optimizer, epoch)
    process = tqdm(train_loader)
    optimizer.zero_grad()
    n = 0
    for batch_idx, (data, label, index) in enumerate(process):
#         data = data[:, :, :, :, 0] - data[:, :, :, :, 1]
#         data = data.unsqueeze(-1)
        data, label = data.float().cuda(output_device), label.long().cuda(output_device)
    
#         scheduler(optimizer, batch_idx, epoch)
    
        output = model(data)
        loss = criterion(output, label) / subdivision
        loss.backward()
        n += 1
        if n == subdivision:
            n = 0
            optimizer.step()
            optimizer.zero_grad()
        loss_value += loss.item()
    print("Mean training loss: {:>7.3f}".format(loss_value / len(train_loader) * subdivision), flush=True)
    
    model.eval()
    process = tqdm(test_loader)
    acc = 0.
    with torch.no_grad():
        for batch_idx, (data, label, index) in enumerate(process):
#             data = data[:, :, :, :, 0] - data[:, :, :, :, 1]
#             data = data.unsqueeze(-1)
            data, label = data.float().cuda(output_device), label.long().cuda(output_device)
            output = model(data)
            _, predict_label = torch.max(output.data, 1)
            acc += (predict_label == label.data).sum()
        
        acc = acc / len(test_loader.dataset)
        print("Test acc: {:>7.2f}%".format(acc * 100.), flush=True)
        
        if acc > best_acc:
            best_acc = acc
            state_dict = model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, arg.model_saved_name + '.pt')

print(best_acc)