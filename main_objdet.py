import argparse
import collections

import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import wandb

import models.retinanet as rt
from xview_dataset import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer, Padder
from torch.utils.data import DataLoader

# from retinanet import coco_eval
from objdet_util import csv_eval

####
#import torchvision.models as models
####

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

## CODEBASE CHANGES:
## 1) Line 371 xview_dataset.py -- don't pad unnecessarily. want to maintain 32x32
## 2) Retinanet and Anchors changes for sat_resnet50 (within if) - line 164 retinanet.py and line 12 anchors.py

## TODO:
## 1) Might want to change optimizer to have different lr for different parts of model like main_semseg.py

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', default='csv', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', default='/atlas/u/kayush/winter2020/jigsaw/pytorch-retinanet/retinanet/train_annots.csv', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', default='/atlas/u/kayush/winter2020/jigsaw/pytorch-retinanet/retinanet/class_list.csv', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', default='/atlas/u/kayush/winter2020/jigsaw/pytorch-retinanet/retinanet/val_annots.csv', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=200)

    parser.add_argument('--pretrained', default='checkpoints/joint_moco_sat_resnet50_lr=0.00015_bs=512_rgb-r=50_sentinel-r=50_rc=32_joint=either_ddb/checkpoint_0199.pth.tar', type=str, help='path to moco pretrained checkpoint')
    parser.add_argument('--pretrained-id', default='joint-200', type=str, help='Pretrained ID')
    parser.add_argument('--savepath', default='', type=str, help='Moco Augmentation Type')
    parser.add_argument('--sat_resnet', '-sr', action='store_true', help='Use smaller version of resnet?')
    parser.add_argument('--num_bands', default=16, type=int, help='Number of bands input in image')
    parser.add_argument('--batch-size', '-bs', default=1024, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
    parser.add_argument('--finetune', '-ft', action='store_true', help='finetune old weights')
    parser.add_argument('--min-side', type=int, default=32, help='min side to resize')
    parser.add_argument('--max-side', type=int, default=32, help='max side to resize')
    parser.add_argument('--print-freq', '-p', type=int, default=1, help='print frequency')

    parser = parser.parse_args(args)
    
    ## Set up some housekeeping
    logfile = f'mocoobjdet_lr={parser.lr}_bs={parser.batch_size}'
    if parser.pretrained:
        logfile += f'_pt={parser.pretrained_id}'
    if parser.finetune:
        logfile += f'_ft'
    parser.savepath = f'checkpoints/{logfile}/'
    #if not os.path.exists(parser.savepath):
    #    os.makedirs(parser.savepath, exist_ok=True)

    wandb.init(
        name=logfile,
        project='moco-v3',
        config=vars(parser),
        entity='ssl-satellites')

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer(min_side=parser.min_side, max_side=parser.max_side), Padder(parser.num_bands-3)]))
        print('==> Train dataset ready!')

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer(min_side=parser.min_side, max_side=parser.max_side), Padder(parser.num_bands-3)]))
            print('==> Val dataset ready!')

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)
    print('==> Train dataloader ready!')

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=parser.batch_size//2, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)
        print('==> Val dataloader ready!')

    if parser.depth == 18:
        retinanet = rt.resnet18(num_classes=dataset_train.num_classes(), pretrained=False, num_bands=parser.num_bands, sat_resnet=parser.sat_resnet)
    elif parser.depth == 34:
        retinanet = rt.resnet34(num_classes=dataset_train.num_classes(), pretrained=False, num_bands=parser.num_bands, sat_resnet=parser.sat_resnet)
    elif parser.depth == 50:
        retinanet = rt.resnet50(num_classes=dataset_train.num_classes(), pretrained=False, num_bands=parser.num_bands, sat_resnet=parser.sat_resnet)
    elif parser.depth == 101:
        retinanet = rt.resnet101(num_classes=dataset_train.num_classes(), pretrained=False, num_bands=parser.num_bands, sat_resnet=parser.sat_resnet)
    elif parser.depth == 152:
        retinanet = rt.resnet152(num_classes=dataset_train.num_classes(), pretrained=False, num_bands=parser.num_bands, sat_resnet=parser.sat_resnet)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
        
    modules_old = [retinanet.conv1, retinanet.bn1, retinanet.layer1, retinanet.layer2, retinanet.layer3, retinanet.layer4]
    
    if not parser.finetune:
        for module in modules_old:
            for param in module.parameters():
                param.requires_grad = False
            
    print('==> Model ready!')

    #####
    # create model
    # retinanet = models.__dict__['resnet50']()
    
    # print(retinanet.state_dict().keys(), retinanet.state_dict()['conv1.weight'].shape)

    # freeze all layers but the last fc
    # for name, param in model.named_parameters():
    #     if name not in ['fc.weight', 'fc.bias']:
    #         param.requires_grad = False

    # init the fc layer
    # retinanet.fc = nn.Linear(retinanet.fc.weight.size(1), dataset_train.num_classes())  # 224*224
    # retinanet.fc.weight.data.normal_(mean=0.0, std=0.01)
    # retinanet.fc.bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    # parser.pretrained = '/atlas/u/kayush/winter2020/jigsaw/moco_sat/moco_code/ckpt/fmow/resnet50/cpc_500/lr-0.03_bs-256_t-0.02_mocodim-128_temporal_224_exactly_same_as_32x32_with_add_transform/checkpoint_0050.pth.tar'
    if os.path.isfile(parser.pretrained):
        print("=> loading checkpoint '{}'".format(parser.pretrained))
        checkpoint = torch.load(parser.pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.fc'):
                # remove prefix
                new_k = k.replace('module.base_encoder.', '').replace('shortcut', 'downsample')
                state_dict[new_k] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = retinanet.load_state_dict(state_dict, strict=False)
        assert msg.unexpected_keys == [], \
            f"Model not loading weights correctly. There are some unused weights: {msg.unexpected_keys}"

        print("=> loaded pre-trained model '{}'".format(parser.pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(parser.pretrained))

    #####

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet, device_ids=[0,1,2,3]).cuda()
        # retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()
    # for layer in retinanet.module.modules():
    #     if isinstance(layer, nn.BatchNorm2d):
    #         layer.eval() 

    print('Num training images: {}'.format(len(dataset_train)))

    print('Num of classes: {}'.format(dataset_train.num_classes()))
    # sys.exit()
    for epoch_num in range(1, parser.epochs+1):

        retinanet.train()
        retinanet.module.freeze_bn()
        # for layer in retinanet.module.modules():
        #     if isinstance(layer, nn.BatchNorm2d):
        #         layer.eval() 

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            
            optimizer.zero_grad()

            if torch.cuda.is_available():
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
            else:
                classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            optimizer.step()

            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))
            
            mean_loss = np.mean(loss_hist)
            
            if (iter_num+1) % parser.print_freq == 0:
                print('Epoch: {} | Iteration: {}/{} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num+1, len(dataloader_train), float(classification_loss), float(regression_loss), mean_loss))
                wandb.log({'training/loss': mean_loss,
                          'training/classification_loss': float(classification_loss),
                          'training/regression_loss': float(regression_loss)})

            del classification_loss
            del regression_loss

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, retinanet)
            
            mAP = np.mean([x[0] for x in mAP.values()])
            wandb.log({'evaluation/MAP': mAP})
            print(f'####### MAP: {mAP} #########')
            
        scheduler.step(np.mean(epoch_loss))
        
        ## NOT SAVING NO MEMORY ON ATLAS
        #filename = parser.savepath + '/train_epoch_' + str(epoch_num) + '.pth'
        #torch.save({'epoch': epoch_num, 'state_dict': retinanet.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, filename)

    retinanet.eval()

    #torch.save(retinanet, 'ckpt/model_final.pt')


if __name__ == '__main__':
    main()