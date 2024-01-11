# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import mrcfile
import pickle
import torch.multiprocessing as mp
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae
from engine_pretrain import train_one_epoch#, eval_one_epoch

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, 
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int) 
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.5, type=float, # 0.75 --> 0.5
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--output_dir', default='/data/output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/data/output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--method', default='0', type=int,
                        help='dataset method')
    return parser

class MyCustomDataset(Dataset):
    def __init__(self, path, args):
        self.data_path = path
        self.transform_train = transforms.Compose([
            transforms.Resize(args.input_size, antialias=True),
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3, antialias=True),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            ])

    def __len__(self):
        return len(self.data_path) # Return the size of dataset
    
    def __getitem__(self, index):
        data_dir, indx = self.data_path[index] 
        data = mrcfile.read(data_dir)
        data = torch.from_numpy(data)
        data = torch.unsqueeze(data, 1)
        data_std = torch.std(data)
        data_mean = torch.mean(data)
        data = (data - data_mean) / data_std
        data = data[indx,:,:]
        data = self.transform_train(data)
        return data

class MyCustomDataset2(Dataset):
    def __init__(self, args):
        self.transform_train = transforms.Compose([
            transforms.Resize(args.input_size, antialias=True),
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3, antialias=True),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            ])
        sample_num = np.arange(1,601)
        sample_text = [f"{i:03}" for i in sample_num]
        data_path = ['/denoised/MRC_0601/'+ s +'_particles_shiny_nb50_new.mrcs' for s in sample_text]

        sample_num = np.arange(1,482)
        sample_text = [f"{i:03}" for i in sample_num]
        data_path = data_path + ['/denoised/MRC_1901/'+ s +'_particles_shiny_nb50_new.mrcs' for s in sample_text]
        np.random.shuffle(data_path)
        train_size = int(0.8 * len(data_path))
        data_path = data_path[:train_size] 
        self.data_path = []
        for d in data_path:
            self.data_path.append([d,0])
            self.data_path.append([d,1])
            #self.data_path.append([d,2])
    
    def __len__(self):
        return len(self.data_path) # Return the size of dataset
    
    def __getitem__(self, index):
        data_dir, id = self.data_path[index] 
        data = mrcfile.read(data_dir)
        data = torch.from_numpy(data)
        data_std = torch.std(data)
        data_mean = torch.mean(data)
        data = (data - data_mean) / data_std
        if id == 0:
            data = data[:data.size(0)//2]
        elif id == 1:
            data = data[data.size(0)//2:]
        data = self.transform_train(data)
        return data
    
def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    #device = torch.device(args.device)
    device = args.gpu

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    print('seed: ',seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    with open("/data/train_data_path_list", "rb") as fp:   # Unpickling
        train_data_path = pickle.load(fp)
    #with open("/data/validation_data_path_list", "rb") as fp:   # Unpickling
    #    evalidation_data_path = pickle.load(fp)
    
    #train_data_path = train_data_path[:100]
    #evalidation_data_path = evalidation_data_path[:100]
    if args.method == 0:
        dataset_train = MyCustomDataset(train_data_path, args)  
    else:
        dataset_train = MyCustomDataset2(args)

    #dataset_eval = MyCustomDataset(evalidation_data_path, args)  
    print('dataset for train created with len = ', len(dataset_train))
    #print('dataset for eval created with len = ', len(dataset_eval))
    if args.distributed: #True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        print("num_tasks:", num_tasks)
        print("global_rank:", global_rank)
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        #sampler_eval = torch.utils.data.DistributedSampler(
        #    dataset_eval, num_replicas=num_tasks, rank=global_rank, shuffle=False
        #)
        #print("Sampler_eval = %s" % str(sampler_eval))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        #sampler_eval = torch.utils.data.RandomSampler(dataset_eval)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.method == 0:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False, # changed to False
        )
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False, # changed to False
        )
    #data_loader_eval = torch.utils.data.DataLoader(
    #    dataset_eval, sampler=sampler_eval,
    #    batch_size=args.batch_size,
    #    num_workers=args.num_workers,
    #    pin_memory=args.pin_mem,
    #    drop_last=True,
    #    shuffle=False,
    #)
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    print("world_size: ", misc.get_world_size())
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False) # changed to False
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            #data_loader_eval.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        #eval_state = eval_one_epoch(
        #    model, data_loader_eval,
        #    optimizer, device, epoch,
        #    log_writer=log_writer,
        #    args=args
        #)
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch
                )

        log_stats_train = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        #log_stats_eval = {**{f'eval_{k}': v for k, v in eval_state.items()},
        #                'epoch': epoch,}
        

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats_train) + "\n")
                #f.write(json.dumps(log_stats_eval) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)