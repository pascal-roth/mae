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
from self_sup_seg.third_party.mae.models_mae import MaskedAutoencoderViT
from self_sup_seg.third_party.mae.vicreg.utils import off_diagonal, FullGatherLayer


class MaskedAutoencoderVicReg(MaskedAutoencoderViT):
    """
    Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=False,
                 mask_ratio=0.75,
                 weight_decay=0.05,
                 lr=1e-3,
                 min_lr=0,
                 warmup_epochs=40,
                 total_train_epochs: int = 400,
                 weight_mae_loss: float = 0.75,
                 weight_vic_loss: float = 0.25,
                 sim_coeff: float = 25.0,
                 std_coeff: float = 25.0,
                 cov_coeff: float = 1.0,
                 batch_size: int = 64):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads, decoder_embed_dim,
                         decoder_depth, decoder_num_heads, mlp_ratio, norm_layer, norm_pix_loss, mask_ratio,
                         weight_decay, lr, min_lr, warmup_epochs, total_train_epochs)

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
        # x.shape = [BATCH_SIZE, NUMBER_PATCHES, EMBED_DIM]
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
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        return pred, mask, latent

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        rand_mask_int = torch.randint(low=0, high=10000, size=(1,))
        torch.manual_seed(rand_mask_int)  # use torch.manual_seed to ensure that for both augmentations the same mask is produced
        pred_1, mask_1, embed_1 = self.forward(imgs[0])
        torch.manual_seed(rand_mask_int)
        pred_2, mask_2, embed_2 = self.forward(imgs[1])
        loss_mae_1 = self.forward_loss(imgs[0], pred_1, mask_1)
        loss_mae_2 = self.forward_loss(imgs[1], pred_2, mask_2)
        loss_vic = self.forward_vicloss(embed_1, embed_2)
        loss = self.weight_mae_loss * (loss_mae_1 + loss_mae_2) + self.weight_vic_loss * loss_vic
        self.log('vic_train_loss', loss_vic, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('mae1_train_loss', loss_mae_1, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('mae2_train_loss', loss_mae_2, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        rand_mask_int = torch.randint(low=0, high=10000, size=(1,))
        torch.manual_seed(rand_mask_int)
        pred_1, mask_1, embed_1 = self.forward(imgs[0])
        torch.manual_seed(rand_mask_int)
        pred_2, mask_2, embed_2 = self.forward(imgs[1])
        loss_mae_1 = self.forward_loss(imgs[0], pred_1, mask_1)
        loss_mae_2 = self.forward_loss(imgs[1], pred_2, mask_2)
        loss_vic = self.forward_vicloss(embed_1, embed_2)
        loss = self.weight_mae_loss * (loss_mae_1 + loss_mae_2) + self.weight_vic_loss * loss_vic
        self.log('vic_val_loss', loss_vic, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('mae1_val_loss', loss_mae_1, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('mae2_val_loss', loss_mae_2, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        param_groups = optim_factory.add_weight_decay(self, self.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=(0.9, 0.95))
        scheduler = WarmupCosLRScheduler(optimizer, init_lr=self.lr, warmup_epochs=self.warumup_epochs,
                                         min_lr=self.min_lr, total_epochs=self.total_train_epochs)
        return [optimizer], [scheduler]

    def lr_scheduler_step(
        self,
        scheduler: WarmupCosLRScheduler,
        optimizer_idx: int,
        metric: Optional[Any],
    ) -> None:
        scheduler.step(epoch=self.current_epoch)


# def Projector(mlp, embedding):
#     mlp_spec = f"{embedding}-{mlp}"
#     layers = []
#     f = list(map(int, mlp_spec.split("-")))
#     for i in range(len(f) - 2):
#         layers.append(nn.Linear(f[i], f[i + 1]))
#         layers.append(nn.BatchNorm1d(f[i + 1]))
#         layers.append(nn.ReLU(True))
#     layers.append(nn.Linear(f[-2], f[-1], bias=False))
#     return nn.Sequential(*layers)


def mae_vit_base_patch16_vic(**kwargs):
    model = MaskedAutoencoderVicReg(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_vic(**kwargs):
    model = MaskedAutoencoderVicReg(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_vic(**kwargs):
    model = MaskedAutoencoderVicReg(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
# mae_vit_base_patch16 = mae_vit_base_patch16_vic  # decoder: 512 dim, 8 blocks
# mae_vit_large_patch16 = mae_vit_large_patch16_vic  # decoder: 512 dim, 8 blocks
# mae_vit_huge_patch14 = mae_vit_huge_patch14_vic  # decoder: 512 dim, 8 blocks
