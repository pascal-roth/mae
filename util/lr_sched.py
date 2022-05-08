# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


# def adjust_learning_rate(optimizer, epoch, args):
#     """Decay the learning rate with half-cycle cosine after warmup"""
#     if epoch < args.warmup_epochs:
#         lr = args.lr * epoch / args.warmup_epochs
#     else:
#         lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
#             (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
#     for param_group in optimizer.param_groups:
#         if "lr_scale" in param_group:
#             param_group["lr"] = lr * param_group["lr_scale"]
#         else:
#             param_group["lr"] = lr
#     return lr



class LearningRateScheduler(_LRScheduler):
    r"""
    Provides inteface of learning rate scheduler.
    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, optimizer, init_lr):
        self.optimizer = optimizer
        self.init_lr = init_lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']


class WarmupCosLRScheduler(LearningRateScheduler):
    """
    Warmup learning rate until `total_steps`
    Args:
        optimizer (Optimizer): wrapped optimizer.
        configs (DictConfig): configuration set.
    """
    def __init__(
            self,
            optimizer: Optimizer,
            init_lr: float,
            min_lr: float,
            warmup_epochs: int
    ) -> None:
        super(WarmupCosLRScheduler, self).__init__(optimizer, init_lr, min_lr, warmup_epochs)
        self.update_steps = 1
        self.lr = init_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs

    def step(self, epoch, val_loss: Optional[torch.FloatTensor] = None):
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs
        else:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * \
                 (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)))
        self.set_lr(self.optimizer, lr)
        self.lr = lr
        self.update_steps += 1
        return lr
