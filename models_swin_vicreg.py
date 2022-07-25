from functools import partial
from typing import Optional, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
from timm.models.vision_transformer import PatchEmbed, Block

from self_sup_seg.third_party.mae.util.pos_embed import get_2d_sincos_pos_embed
from self_sup_seg.third_party.mae.util.lr_sched import WarmupCosLRScheduler
from self_sup_seg.third_party.mae.models_swin import MaskedAutoencoderSwin
from self_sup_seg.third_party.mae.vicreg.utils import off_diagonal, FullGatherLayer


class MaskedAutoencoderSwinVICReg(MaskedAutoencoderSwin):
    """
    Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=4, 
                 in_chans=3,  
                 embed_dim=96,
                 depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24],  
                 mlp_ratio=4., 
                 norm_layer=nn.LayerNorm, 
                 norm_pix_loss=False,
                 mask_ratio=0.75, 
                 weight_decay=0.05, 
                 lr=1e-3,
                 min_lr=0, 
                 warmup_epochs=40,
                 total_train_epochs: int = 800, 
                 decoder: str = 'DecoderFPN',
                 area_mask: bool = False,
                 weight_mae_loss: float = 0.75,
                 weight_vic_loss: float = 0.25,
                 sim_coeff: float = 25.0,
                 std_coeff: float = 25.0,
                 cov_coeff: float = 1.0,
                 batch_size: int = 64):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depths, num_heads, mlp_ratio, norm_layer, 
                         norm_pix_loss, mask_ratio, weight_decay, lr, min_lr, warmup_epochs, total_train_epochs,
                         decoder, area_mask)

        self.weight_mae_loss: float = weight_mae_loss
        self.weight_vic_loss: float = weight_vic_loss
        self.sim_coeff: float = sim_coeff
        self.std_coeff: float = std_coeff
        self.cov_coeff: float = cov_coeff
        self.batch_size: int = batch_size
        assert self.weight_mae_loss + self.weight_vic_loss == 1, 'Loss weights have to add up to 1'

        self.embed_dim: int = embed_dim

    def forward_vicloss(self, x, y):
        """
        embedding of x_vector
        """
        # x.shape = [BATCH_SIZE, EMBED_DIM, PATCH_W, PATCH_H]
        
        # bring into shape [BATCH_SIZE, NUMBER_PATCHES, EMBED_DIM]
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
        x = x.moveaxis(1,2)
        y = y.reshape(y.shape[0], y.shape[1], y.shape[2]*y.shape[3])
        y = y.moveaxis(1,2) 

        # get VIC losses
        repr_loss = torch.nn.functional.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)  # only if distributed training is activated
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=(0,1))  # mean along each dimension, i.e. mean over batch and patches
        y = y - y.mean(dim=(0,1))

        std_x = torch.sqrt(x.var(dim=(0,1)) + 0.0001)  # variance along each dimension, i.e. var over batch and patches
        std_y = torch.sqrt(y.var(dim=(0,1)) + 0.0001)
        std_loss = torch.mean(torch.nn.functional.relu(1 - std_x)) / 2 + torch.mean(torch.nn.functional.relu(1 - std_y)) / 2

        flat_x = torch.flatten(x, start_dim=0, end_dim=1)  # flatten input vecor, s.t. flat_x.shape = [BATCH_SIZE * NUMBER_PATCHES, EMBED_DIM]
        flat_y = torch.flatten(y, start_dim=0, end_dim=1)
        cov_x = (flat_x.T @ flat_x) / (self.batch_size + x.shape[1] - 1)  # batches and patches are traited equally, thus here normalized with both
        cov_y = (flat_y.T @ flat_y) / (self.batch_size + x.shape[1] - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.embed_dim
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.embed_dim)

        loss = (self.sim_coeff * repr_loss
                + self.std_coeff * std_loss
                + self.cov_coeff * cov_loss
        ) 
        return loss

    def forward(self, imgs):
        latent, mask = self.forward_encoder(imgs)
        pred = self.decoder(latent)  # [N, L, p*p*3]
        return pred, mask, latent

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        # rand_mask_int = torch.randint(low=0, high=10000, size=(1,))
        # torch.manual_seed(rand_mask_int)  # use torch.manual_seed to ensure that for both augmentations the same mask is produced
        pred_1, mask_1, embed_1 = self.forward(imgs[0])
        # torch.manual_seed(rand_mask_int)
        pred_2, mask_2, embed_2 = self.forward(imgs[1])
        loss_mae_1 = self.forward_loss(imgs[0], pred_1, mask_1)
        loss_mae_2 = self.forward_loss(imgs[1], pred_2, mask_2)
        loss_vic = self.forward_vicloss(embed_1[f'res{self.out_indices[-1] + 2}'], embed_2[f'res{self.out_indices[-1]+2}'])
        loss = self.weight_mae_loss * (loss_mae_1 + loss_mae_2) + self.weight_vic_loss * loss_vic
        self.log('vic_train_loss', loss_vic, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('mae1_train_loss', loss_mae_1, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('mae2_train_loss', loss_mae_2, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        # rand_mask_int = torch.randint(low=0, high=10000, size=(1,))
        # torch.manual_seed(rand_mask_int)
        pred_1, mask_1, embed_1 = self.forward(imgs[0])
        # torch.manual_seed(rand_mask_int)
        pred_2, mask_2, embed_2 = self.forward(imgs[1])
        loss_mae_1 = self.forward_loss(imgs[0], pred_1, mask_1)
        loss_mae_2 = self.forward_loss(imgs[1], pred_2, mask_2)
        loss_vic = self.forward_vicloss(embed_1[f'res{self.out_indices[-1] + 2}'], embed_2[f'res{self.out_indices[-1]+2}'])
        loss = self.weight_mae_loss * (loss_mae_1 + loss_mae_2) + self.weight_vic_loss * loss_vic
        self.log('vic_val_loss', loss_vic, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('mae1_val_loss', loss_mae_1, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('mae2_val_loss', loss_mae_2, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

def mae_swin_t_vic(**kwargs):
    pretrained_weights = kwargs.pop('pretrain_path', None)
    model = MaskedAutoencoderSwinVICReg(
        patch_size=4, in_chans=3,  # swin defaults: patch size: 4
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],  # swin defaults: embed_dim=96
        mlp_ratio=4.0, **kwargs)

    if pretrained_weights:
        try:
            msg = model.load_state_dict(torch.load(pretrained_weights, map_location=model.device)['model'], strict=False)
        except KeyError:
            msg = model.load_state_dict(torch.load(pretrained_weights, map_location=model.device), strict=False)
        print(f"PRE-TRAINED WEIGHTS LOADED FROM {pretrained_weights}")
        print(msg)

    # attn_drop_rate = 0.0
    # ape = False
    # drop_path_rate = 0.3
    # drop_rate = 0.0
    # out features = ['res2', 'res3', 'res4', 'res5']
    # qkv_bias = True
    # qk_scale = None
    # window_size = 7
    return model
