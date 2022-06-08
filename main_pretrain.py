# Modified by Pascal Roth
# Email: roth.pascal@outlook.de

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
import os
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import self_sup_seg.third_party.mae.models_mae as models_mae
import self_sup_seg.third_party.mae.models_mae_vicreg as models_mae_vicreg
import self_sup_seg.third_party.mae.models_swin as models_swin
import self_sup_seg.third_party.mae.models_swin_vicreg as models_swin_vicreg
from self_sup_seg.third_party.mae.vicreg.augmentation import TrainTransform

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.plugins.precision.mixed import MixedPrecisionPlugin

# import parameters
from self_sup_seg.third_party.mae.params import IMAGE_MEAN, IMAGE_STD


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='mae_swin_t', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--pretrain', default='./self_sup_seg/models/mae_pytorch/swin_tiny_patch4_window7_224.pth', 
                        type=str, help='path to pre-trained model weight (only use weights from original repo)')

    # SWIN Model parameters
    parser.add_argument('--swin', action='store_true',
                        help='decide if swin architecture should be used')
    parser.add_argument('--area_mask', action='store_true', 
                        help='For masking use a patch size 4 times larger than the swin patch size (masking same as for ViT)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Training parameters
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,  # implemented as pytorch lightning trainer flag
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory '
                             'constraints)')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save checkpoints')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where model wandb and tensorboard files are saved')
    parser.add_argument('--devices', default=1,
                        help='number of devices to use')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--ckpt_path', default=None,
                        help='ckpt_path to resume training from')

    # Wandb Parameters
    parser.add_argument('--wb-name', type=str, default='mae',
                        help='Run name for Weights and Biases')
    parser.add_argument('--wb-project', type=str, default='ssl_pan_seg',
                        help='Project name for Weights and Biases')
    parser.add_argument('--wb-entity', type=str, default="rsl_ssl_pan_seh",
                        help='Entity name for Weights and Biases')

    # Dataoader parameters
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--data_path', default='./self_sup_seg/data/dataset_unlabeled', type=str,
                        help='dataset path')
    # parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
    #                     help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU. Means load'
                             'all training data into RAM (?)')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--pixel_mean', default=IMAGE_MEAN,
                        help='Inputs are normalized with this mean (default is the one from params.py)')
    parser.add_argument('--pixel_std', default=IMAGE_STD,
                        help='Inputs are normalized with this std (default is the one from params.py)')

    # Variance-Invariance-Covariance Loss Parameters
    parser.add_argument('--vic', action='store_true',
                        help='Activate VicReg Loss')
    parser.add_argument('--vic-aug', action='store_true',
                        help='activate the additional augmentations presented in VICReg')
    parser.add_argument("--sim-coeff", type=float, default=10.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=10.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=0.5,
                        help='Covariance regularization loss coefficient')
    parser.add_argument("--weight-mae", type=float, default=0.5,
                        help='Weight for MAE loss')
    parser.add_argument("--weight-vic", type=float, default=0.5,
                        help='Weight for VIC loss')
    return parser


def main(args):
    # seed everything
    seed_everything(args.seed, workers=True)

    # simple augmentation and data loading
    if args.vic:
        transform_train = TrainTransform(args.input_size, args.vic_aug)
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.pixel_mean, std=args.pixel_std)])

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    # TODO: potentially change val set transforms similar to the one used in main_finetune
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_train)
    print(dataset_train)
    print(dataset_val)

    if False:  # TODO: for distributed training, still have to test it (and introduce args.distributed):
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=args.devices, shuffle=True  # , rank=global_rank
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # define LOGGER
    wandb_logger = WandbLogger(
        name=args.wb_name,
        project=args.wb_project,
        entity=args.wb_entity,
        save_dir=args.log_dir,
    )
    tb_logger = TensorBoardLogger(
        name="tb",
        version="",
        save_dir=args.log_dir,
    )

    # define CALLBACKS
    checkpoint_local_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        save_last=True,
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # define the model
    if args.vic and args.swin:
        model = models_swin_vicreg.__dict__[args.model + '_vic'](norm_pix_loss=args.norm_pix_loss, mask_ratio=args.mask_ratio,
                                                                 weight_decay=args.weight_decay, lr=args.lr, min_lr=args.min_lr,
                                                                 warmup_epochs=args.warmup_epochs, img_size=args.input_size,
                                                                 total_train_epochs=args.epochs, pretrain_path=args.pretrain, 
                                                                 area_mask=args.area_mask, weight_mae_loss=args.weight_mae,
                                                                 weight_vic_loss=args.weight_vic, sim_coeff=args.sim_coeff, 
                                                                 std_coeff=args.std_coeff, cov_coeff=args.cov_coeff)
    elif args.vic:
        model = models_mae_vicreg.__dict__[args.model + '_vic'](norm_pix_loss=args.norm_pix_loss, batch_size=args.batch_size,
                                                            mask_ratio=args.mask_ratio, min_lr=args.min_lr,
                                                            weight_decay=args.weight_decay, lr=args.lr,
                                                            warmup_epochs=args.warmup_epochs, img_size=args.input_size,
                                                            total_train_epochs=args.epochs, weight_mae_loss=args.weight_mae,
                                                            weight_vic_loss=args.weight_vic, sim_coeff=args.sim_coeff, 
                                                            std_coeff=args.std_coeff, cov_coeff=args.cov_coeff)
    elif args.swin:
        model = models_swin.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, mask_ratio=args.mask_ratio,
                                                 weight_decay=args.weight_decay, lr=args.lr, min_lr=args.min_lr,
                                                 warmup_epochs=args.warmup_epochs, img_size=args.input_size,
                                                 total_train_epochs=args.epochs, pretrain_path=args.pretrain, 
                                                 area_mask=args.area_mask)
    else:
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, mask_ratio=args.mask_ratio,
                                                weight_decay=args.weight_decay, lr=args.lr, min_lr=args.min_lr,
                                                warmup_epochs=args.warmup_epochs, img_size=args.input_size,
                                                total_train_epochs=args.epochs, pretrain_path=args.pretrain)

    trainer = Trainer(accumulate_grad_batches=args.accum_iter, gradient_clip_val=0,
                      logger=[wandb_logger, tb_logger], 
                      callbacks=[checkpoint_local_callback, lr_monitor],
                      max_epochs=args.epochs,
                      strategy="ddp",
                      accelerator="gpu", devices=args.devices,
                      # plugins=[MixedPrecisionPlugin()], precision=32,  # same as doing the loss_scaler
                      )

    # trainer.fit(model, train_dataloaders=data_loader_train, ckpt_path=args.ckpt_path)
    trainer.fit(model, train_dataloaders=data_loader_train, val_dataloaders=data_loader_val, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(args)
    main(args)
