#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import numpy as np
import builtins
import math
import os
import random
import shutil
import time
import pickle
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import RandomVerticalFlip
import wandb

import moco.builder
import moco.loader
import moco.optimizer
from fmow_dataloader import fMoWMultibandDataset, fMoWRGBDataset, fMoWJointDataset
import viewmaker_moco
import vits
from moco.sat_resnet import resnet50

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] \
                + ['sat_resnet50'] \
                + torchvision_model_names

parser = argparse.ArgumentParser(description='MoCo Pre-Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset-name', type=str, help='Name of the dataset', default='fmow-joint', choices=['fmow-rgb', 'fmow-multi', 'fmow-joint'])
parser.add_argument('--dont-drop-bands', '-ddb', action='store_true')
parser.add_argument('--viewmaker', '-vm', action='store_true', help='Using viewmaker or not')
parser.add_argument('-a', '--arch', metavar='ARCH', default='sat_resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: sat_resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel') 
parser.add_argument('--lr', '--learning-rate', default=1.5e-4, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:12345', type=str,    
                    help='url used to set up distributed training')        
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--moco-m-cos', action='store_true',
                    help='gradually increase moco momentum to 1 with a '
                         'half-cycle cosine schedule')
parser.add_argument('--moco-t', default=1.0, type=float,
                    help='softmax temperature (default: 1.0)')

# vit specific configs:
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')

# other upgrades
parser.add_argument('--optimizer', default='adamw', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer used (default: lars)')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (default: 0.08)')
parser.add_argument('--sentinel-resize', default=50, type=int)
parser.add_argument('--rgb-resize', default=224, type=int)
parser.add_argument('--crop-size', default=32, type=int)
parser.add_argument('--joint-transform', type=str, choices=['either', 'drop'], default='either')


def main():
    args = parser.parse_args()
    
    if args.joint_transform == 'drop':
        assert args.dont_drop_bands

    if args.arch.startswith('vit'):
        raise NotImplementedError
    if args.viewmaker:
        raise NotImplementedError
    if args.dataset_name == 'fmow-rgb':
        raise NotImplementedError

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # create augmentations
    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733 
    # But use only those usable for multiband images and add some more
    if args.dataset_name == 'fmow-multi':
        channel_stats = pickle.load(open('./fmow-multiband-log-stats.pkl', 'rb'))
        normalize = transforms.Normalize(mean=channel_stats['log_channel_means'],
                                            std=channel_stats['log_channel_stds'])
        
        augmentation1 = [
            transforms.Resize(args.sentinel_resize),         
            transforms.RandomResizedCrop(args.crop_size, scale=(args.crop_min, 1.)),
            #transforms.RandomApply([
            #    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            #], p=0.8),                      # ColorJitter doesn't work on != 3 channels
            #transforms.RandomGrayscale(p=0.2),     # RandomGrayscale doesn't work on !=3 channels
            #transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=1.0),     # need to check this
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),   ## this is new we added
            transforms.RandomRotation(90),     ## this is new we added
            moco.loader.LogTransform(epsilon=1.),  ## this is new we added
            normalize,
            moco.loader.RandomDropBands()  ## this is new we added
        ]

        augmentation2 = [
            transforms.Resize(args.sentinel_resize),        
            transforms.RandomResizedCrop(args.crop_size, scale=(args.crop_min, 1.)),
            #transforms.RandomApply([
            #    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            #], p=0.8),                      # ColorJitter doesn't work on != 3 channels
            #transforms.RandomGrayscale(p=0.2),   # RandomGrayscale doesn't work on !=3 channels
            #transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1),   # doesn't work out of the box -- might need to change implementation
            #transforms.RandomApply([moco.loader.Solarize()], p=0.2),               # also doesn't work out of the box -- might need to change implementation
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),   ## this is new we added
            transforms.RandomRotation(90),     ## this is new we added
            moco.loader.LogTransform(epsilon=1.),  ## this is new we added
            normalize,
            moco.loader.RandomDropBands()  ## this is new we added
        ]

        if args.dont_drop_bands:
            augmentation1 = augmentation1[:-1]
            augmentation2 = augmentation2[:-1]
    
    elif args.dataset_name == 'fmow-joint':
        rgb_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        channel_stats = pickle.load(open('./fmow-multiband-log-stats.pkl', 'rb'))
        sentinel_normalize = transforms.Normalize(mean=channel_stats['log_channel_means'],
                                            std=channel_stats['log_channel_stds'])                   
                                         
        rgb_transforms = [
            transforms.Resize(args.rgb_resize),
            transforms.RandomResizedCrop(args.crop_size, scale=(args.crop_min, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),   ## this is new we added
            transforms.RandomRotation(90),     ## this is new we added
            transforms.ToTensor(),
            rgb_normalize,
        ]
        sentinel_transforms = [
            transforms.Resize(args.sentinel_resize),         
            transforms.RandomResizedCrop(args.crop_size, scale=(args.crop_min, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),   ## this is new we added
            transforms.RandomRotation(90),     ## this is new we added
            moco.loader.LogTransform(epsilon=1.),  ## this is new we added
            sentinel_normalize,
            moco.loader.RandomDropBands()  ## this is new we added
        ]
        if args.dont_drop_bands:
            sentinel_transforms = sentinel_transforms[:-1]
        
    else:
        raise NotImplementedError
    
    # create model
    if args.dataset_name == 'fmow-multi':
        num_bands = 13
    elif args.dataset_name == 'fmow-joint':
        num_bands = 16
    else:
        raise NotImplementedError
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('vit'):
        model = moco.builder.MoCo_ViT(
            partial(vits.__dict__[args.arch], in_chans=num_bands, stop_grad_conv1=args.stop_grad_conv1), arch=args.arch, num_bands=num_bands, dim=args.moco_dim, mlp_dim=args.moco_mlp_dim, T=args.moco_t)
    elif args.arch == 'sat_resnet50':
        model = moco.builder.MoCo_ResNet(
            moco.sat_resnet.resnet50, arch=args.arch, num_bands=num_bands, dim=args.moco_dim, mlp_dim=args.moco_mlp_dim, T=args.moco_t)
    else:
        model = moco.builder.MoCo_ResNet(
            partial(torchvision_models.__dict__[args.arch], zero_init_residual=True), arch=args.arch, num_bands=num_bands, dim=args.moco_dim, mlp_dim=args.moco_mlp_dim, T=args.moco_t)

    # infer learning rate before changing batch size
    args.base_lr = args.lr   ## for logging purposes
    args.lr = args.lr * args.batch_size / 256
    
    ## Set up some housekeeping
    logfile = f'moco_{args.arch}_lr={args.base_lr}_bs={args.batch_size}'
    if args.dataset_name == 'fmow-joint':
        logfile += f'_joint={args.joint_transform}'
    if args.dont_drop_bands:
        logfile += '_ddb'
    if args.viewmaker:
        logfile += '_vm'
    if not os.path.isdir(f'checkpoints/{logfile}/'):
        os.makedirs(f'checkpoints/{logfile}/')
    if not os.path.isdir(f'runs/{logfile}'):
        os.makedirs(f'runs/{logfile}/')

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.total_batchsize = args.batch_size
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm

    if args.optimizer == 'lars':
        optimizer = moco.optimizer.LARS(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
        
    scaler = torch.cuda.amp.GradScaler()
    summary_writer = SummaryWriter(log_dir=f'runs/{logfile}/') if args.rank == 0 else None
    if args.rank == 0:
        wandb.init(
            name=logfile,
            project='moco-v3',
            config=vars(args),
            entity='ssl-satellites')
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    # Data loading code
    traindir = args.data
   
    if args.dataset_name == 'fmow-multi':
        train_dataset = fMoWMultibandDataset(traindir, transforms=moco.loader.TwoCropsTransform(transforms.Compose(augmentation1), 
                                  transforms.Compose(augmentation2)))
    elif args.dataset_name == 'fmow-joint':
        train_dataset = train_dataset = fMoWJointDataset(traindir,
                                        sentinel_transforms=moco.loader.TwoCropsTransform(transforms.Compose(sentinel_transforms), 
                                                                              transforms.Compose(sentinel_transforms)),
                                        rgb_transforms=moco.loader.TwoCropsTransform(transforms.Compose(rgb_transforms),
                                                                         transforms.Compose(rgb_transforms)),
                                        joint_transform=args.joint_transform)
    else:
        raise NotImplementedError
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, scaler, summary_writer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0): # only the first GPU saves checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }, is_best=False, filename=f'checkpoints/{logfile}/checkpoint_%04d.pth.tar' % epoch)

    if args.rank == 0:
        summary_writer.close()

def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    
    ## This is set for without Viewmaker

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    for i, (images, _, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)
        
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            
        ## Setup for using Autocast 
        images[0] = images[0].half()
        images[1] = images[1].half()

        # compute output
        with torch.cuda.amp.autocast(True):
            loss = model(images[0], images[1], moco_m)

        losses.update(loss.item(), images[0].size(0))
        if args.rank == 0:
            summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)
            wandb.log({"loss": loss.item()})

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoints/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


if __name__ == '__main__':
    main()



#python main_moco.py --dataset-name fmow-joint -ddb --joint-transform drop --arch sat_resnet50 -p 10 --moco-m-cos --crop-min=.2 --multiprocessing-distributed --world-size 1 --rank 0 /atlas/u/pliu1/housing_event_pred/data/fmow-sentinel-filtered-csv/train.csv
