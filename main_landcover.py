#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import math
import wandb

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

from landcover_datasets import RandomFlipAndRotateSinglePatch
from landcover_datasets import ClipAndScaleSinglePatch, ToFloatTensorSinglePatch
from landcover_datasets import PatchDataset
from sklearn.preprocessing import LabelEncoder
import numpy as np

from models.sat_resnet import resnet50

import torchvision 

model_names = sorted(name for name in torchvision.models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(torchvision.models.__dict__[name])) + ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + ['sat_resnet50']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset-name', type=str, default='naip', help='dataset name')
parser.add_argument('--tile-dir', type=str, default='/atlas/u/kayush/pix2vec/supervised_50_100/', help='path to tile dir')
parser.add_argument('--split-fn', type=str, default='/atlas/u/kayush/pix2vec/splits.npy', help='path to dataset')
parser.add_argument('--y-fn', type=str, default='/atlas/u/kayush/pix2vec/y_50_100.npy', help='path to labels')
parser.add_argument('--pad-data', type=bool, default=True, help='Pad data to 16 bands?')
parser.add_argument('-a', '--arch', metavar='ARCH', default='sat_resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: sat_resnet50)')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of data loading workers (default: 20)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
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
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

parser.add_argument('--pretrained', default='checkpoints/joint_moco_sat_resnet50_lr=0.00015_bs=512_rgb-r=50_sentinel-r=50_rc=32_joint=either_ddb/checkpoint_0199.pth.tar', type=str, help='path to moco pretrained checkpoint')
parser.add_argument('--pretrained-id', default='joint-200', type=str, help='Pretrained ID')
parser.add_argument('--eval_model', default='', type=str, help='path to eval model')
parser.add_argument('--fully-supervised', '-fs', action='store_true',
                    help='train a fully supervised model from scratch')
parser.add_argument('--finetune', '-ft', action='store_true',
                    help='load the weights and finetune')
#parser.add_argument('--resize', type=int, default=32, help='Resize image to (r,r)')


best_acc1 = 0

def main():
    args = parser.parse_args()
    
    if args.pretrained and args.fully_supervised:
        raise ValueError("Cannot specify both fully supervised and pretrained to be true")
        
    if args.resume:
        raise NotImplementedError
    
    if args.arch.startswith('vit'):
        raise NotImplementedError
        
    if args.evaluate and args.eval_model:
        SEED = 1
        np.random.seed(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)

    if args.seed is not None:
        random.seed(args.seed)
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
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
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
    # create model
    print("=> creating model '{}'".format(args.arch))

    ### UPDATE THE FINAL LAYER AND CHANGE IT TO WHAT WE NEED. 62 CLASSES NOT 1000
    if 'naip' in args.dataset_name:
        output_dim = 66
    else:
        raise NotImplementedError
    if args.arch.startswith('vit'):
        model = vits.__dict__[args.arch]()
        hidden_dim = model.head.weight.shape[1]
        del model.head
        linear_keyword = 'head'
    elif args.arch.startswith('sat_resnet50'):
        model = resnet50()
        hidden_dim = model.fc.weight.shape[1]
        if args.pad_data:
            num_bands = 16
        else:
            num_bands = 3
        del model.fc, model.conv1
        model.fc = nn.Linear(hidden_dim, output_dim)
        model.conv1 = nn.Conv2d(num_bands, 64, kernel_size=3, stride=1, padding=1, bias=False)
        linear_keyword = 'fc'
    else:
        model = torchvision_models.__dict__[args.arch]()
        hidden_dim = model.fc.weight.shape[1]
        if args.pad_data:
            num_bands = 16
        else:
            num_bands = 3
        del model.fc, model.conv1
        model.fc = nn.Linear(hidden_dim, output_dim)
        model.conv1 = nn.Conv2d(num_bands, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        linear_keyword = 'fc'
        
    if not (args.fully_supervised or args.finetune):
        ## this will happen for eval models also but thats fine
        for name, param in model.named_parameters():
            if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
                param.requires_grad = False
    # init the fc layer
    ## this will happen for eval models also but thats fine
    getattr(model, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
    getattr(model, linear_keyword).bias.data.zero_()
        
    if args.evaluate:
        if os.path.isfile(args.eval_model):
            print("=> loading checkpoint '{}'".format(args.eval_model))
            checkpoint = torch.load(args.eval_model, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('module.'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                    del state_dict[k]

            model.load_state_dict(state_dict)
            print("=> loaded eval model '{}'".format(args.eval_model))
        else:
            print("=> no checkpoint found at '{}'".format(args.eval_model))

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
            
    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if not (args.fully_supervised or args.finetune):
        ## this also includes the eval case but thats fine
        assert len(parameters) == 2  # weight, bias
    
    optimizer = torch.optim.Adam(parameters, init_lr,
                                weight_decay=args.weight_decay)

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
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    ################# Land Cover Classification ################
    # Data
    print('==> Prepping data...')
    tile_dir = args.tile_dir
    y_fn = args.y_fn
    splits_fn = args.split_fn

    transform_tr = transforms.Compose([
        #transforms.Resize((args.resize,args.resize)),  ## Maybe we need to resize, ignoring for now
        ClipAndScaleSinglePatch('naip'),
        RandomFlipAndRotateSinglePatch(),
        ToFloatTensorSinglePatch(),
        transforms.Normalize(mean, std)   ## we added this
    ])
    transform_val = transforms.Compose([
        #transforms.Resize((args.resize,args.resize)),
        ClipAndScaleSinglePatch('naip'),
        ToFloatTensorSinglePatch(),
        transforms.Normalize(mean, std)   ## we added this
    ])
    transform_te = transforms.Compose([
        #transforms.Resize((args.resize,args.resize)),
        ClipAndScaleSinglePatch('naip'),
        ToFloatTensorSinglePatch(),
        transforms.Normalize(mean, std)   ## we added this
    ])

    # Encode labels
    y = np.load(y_fn)
    le = LabelEncoder()
    le.fit(y)
    n_classes = len(le.classes_)
    labels = le.transform(y)

    # Getting train/val/test idxs
    splits = np.load(splits_fn)
    idxs_tr = np.where(splits == 0)[0]
    idxs_val = np.where(splits == 1)[0]
    idxs_te = np.where(splits == 2)[0]

    idxs_te = np.concatenate((idxs_val, idxs_te))

    # Setting up Datasets
    train_dataset = PatchDataset(tile_dir, idxs_tr, labels=labels, transform=transform_tr, pad_data=args.pad_data)
    # valset = PatchDataset(tile_dir, idxs_val, labels=labels, transform=transform_val)
    test_dataset = PatchDataset(tile_dir, idxs_te, labels=labels, transform=transform_te, pad_data=args.pad_data)


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)


    ############### Land Cover ##############
    val_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    ##########################################

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    ## Set up some housekeeping
    logfile = f'mocolandcover_lr={args.lr}_bs={args.batch_size}'
    if args.pretrained:
        logfile += f'_pt={args.pretrained_id}'
    if args.fully_supervised:
        logfile += f'_fs'
    if args.finetune:
        logfile += f'_ft'
    #if not os.path.exists(f'checkpoints/{logfile}/'):
    #    os.makedirs(f'checkpoints/{logfile}/', exist_ok=True)
    #if not os.path.exists(f'runs/{logfile}'):
    #    os.makedirs(f'runs/{logfile}/', exist_ok=True)
        
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
        wandb.init(
            name=logfile,
            project='moco-v3',
            config=vars(args),
            entity='ssl-satellites')

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss, train_acc1, train_acc5 = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args, epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
            ## NOT SAVING NO MEMORY ON ATLAS
            #if epoch % 10 == 0 or epoch == args.epochs - 1:
            #    save_checkpoint({
            #        'epoch': epoch + 1,
            #        'arch': args.arch,
            #        'state_dict': model.state_dict(),
            #        'best_acc1': best_acc1,
            #        'optimizer': optimizer.state_dict(),
            #    }, is_best, filename=f'checkpoints/{logfile}/checkpoint_%04d.pth.tar' % epoch)
            if args.pretrained and epoch == args.start_epoch and not (args.fully_supervised or args.finetune):
                sanity_check(model.state_dict(), args.pretrained, linear_keyword)
            wandb.log({"training/loss": loss, "training/acc1": train_acc1, "training/acc5": train_acc5, "val/acc1": acc1, "val/acc5": acc5})


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    if not (args.fully_supervised or args.finetune):
        model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, args, epoch=-1):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def sanity_check(state_dict, pretrained_weights, linear_keyword):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore linear layer
        if '%s.weight' % linear_keyword in k or '%s.bias' % linear_keyword in k:
            continue

        # name in pretrained model
        k_pre = 'module.base_encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.base_encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


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
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()