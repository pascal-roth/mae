# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from typing import Optional, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl

# import timm
#
# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from self_sup_seg.third_party.mae.util.pos_embed import get_2d_sincos_pos_embed
from self_sup_seg.third_party.mae.util.lr_sched import WarmupCosLRScheduler

from self_sup_seg.third_party.mae.swin.swin import BasicLayer, PatchMerging, SwinTransformer
from self_sup_seg.third_party.mae.swin.swin import PatchEmbed as SwinPatchEmbed
import self_sup_seg.third_party.mae.swin.decoders as Decoders


class MaskedAutoencoderSwin(pl.LightningModule):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, 
                 img_size=224, 
                 patch_size=4, 
                 in_chans=3,  
                 embed_dim=96,
                 depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24],  
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
                 total_train_epochs: int = 800, 
                 decoder: str = 'DecoderFPN',
                 area_mask: bool = False
                ) -> None:
        super().__init__()

        # --------------------------------------------------------------------------
        self.mask_ratio: float = mask_ratio
        self.weight_decay: float = weight_decay
        self.lr: float = lr
        self.min_lr: float = min_lr
        self.warumup_epochs: int = warmup_epochs
        self.total_train_epochs: int = total_train_epochs
        self.area_mask: bool = area_mask
        self.area_mask_constant: int = 4
        if self.area_mask:
            assert img_size % self.area_mask_constant == 0, f'For increased patch size masking, the img size has to be dividable by {self.area_mask_constant}'

        self.save_hyperparameters()

        # eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
        #
        # if args.lr is None:  # only base_lr is specified
        #     args.lr = args.blr * eff_batch_size / 256
        #
        # print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        # print("actual lr: %.2e" % args.lr)

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # TODO: difference between timm/ swin patch embedding is an additional norm and transpose step, investigate which works
        # self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        # num_patches = self.patch_embed.num_patches
        self.patch_embed = SwinPatchEmbed(patch_size, in_chans, embed_dim, norm_layer)


        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
        #                               requires_grad=False)  # fixed sin-cos embedding

        # swin model ###################
        window_size = 7
        mlp_ratio = 4.0
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.0
        attn_drop_rate = 0.0
        drop_path_rate = 0.2
        self.ape = False
        patch_norm = True
        self.out_indices = (0, 1, 2, 3)
        frozen_stages = -1
        use_checkpoint = False
        self.num_layers = len(depths)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # add a norm layer for each output
        for i_layer in self.out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        # self.layers = nn.ModuleList([
        #     Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(depth)])
        # self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder = Decoders.__dict__[decoder](self.out_indices, self.num_layers, embed_dim)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # TODO: make count of patches global
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
        #                                     cls_token=True)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
        #                                             int(self.patch_embed.num_patches ** .5), cls_token=True)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # TODO: evaluate if difference in inititalization is significant
            # we use xavier_uniform following official JAX ViT:
            # torch.nn.init.xavier_uniform_(m.weight)
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
  
        # generate the binary mask: 0 is keep, 1 is remove
        if self.area_mask:
            len_keep = int(L/self.area_mask_constant * (1 - self.mask_ratio))
            mask = torch.ones([int(N/self.area_mask_constant), int(L/self.area_mask_constant)], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
        else:
            len_keep = int(L * (1 - self.mask_ratio))
            mask = torch.ones([int(N/4), int(L/4)], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # unsqueeze mask
        mask_extend = mask.unsqueeze(-1).repeat(1, 1, D)
        
        # mask x
        x_masked = x.masked_fill(mask_extend == 1, 0)

        return x_masked, mask

    def forward_encoder(self, x):
        # embed patches
        # x = self.patch_embed(x)

        # swin embedding
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        # TODO: check if absolute poisition encoder should be used someday, if so, this step has to be changed
        if self.ape:
            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:, :]  # position embedding from mae (maybe change to m2f swin ape)  
       
        x = x.flatten(2).transpose(1,2)

        # masking: length -> length * self.mask_ratio
        x, mask = self.random_masking(x)

        # append cls token
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)

        # apply Swin Transformer layers  # TODO: change to just apply complete encoder (similiar to m2f)
        outs = {}
        for i in range(self.num_layers):
            blk = self.layers[i]
            x_out, H, W, x, Wh, Ww = blk(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs["res{}".format(i + 2)] = out

        # m2f output is dict for an image of size 1024 (in general every image size can be passed through swin!!!): 
        # ('res2', torch.Size([1, 96, 256, 256])), 
        # ('res3', torch.Size([1, 192, 128, 128])), 
        # ('res4', torch.Size([1, 384, 64, 64])), 
        # ('res5', torch.Size([1, 768, 32, 32]))
        # ----> have to work with tokens in the input
        # swin output for training image of size 244x244
        # ('res2', torch.Size([1, 96, 56, 56])), 
        # ('res3', torch.Size([1, 192, 28, 28])), 
        # ('res4', torch.Size([1, 384, 14, 14])), 
        # ('res5', torch.Size([1, 768, 7, 7]))


        return outs, mask

    # def forward_decoder(self, outs):  # TODO: change completely, s.t. it can receive multiple masks
    #     # embed tokens
    #     x = self.decoder_embed(x)

    #     # # append mask tokens to sequence
    #     # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    #     # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
    #     # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    #     # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

    #     # add pos embed
    #     # x = x + self.decoder_pos_embed

    #     # apply Transformer layers
    #     for blk in self.decoder_blocks:
    #         x = blk(x)
    #     x = self.decoder_norm(x)

    #     # predictor projection
    #     x = self.decoder_pred(x)

    #     # remove cls token
    #     x = x[:, 1:, :]

    #     return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - imgs) ** 2
        loss_patched = self.patchify(loss)
        loss_patched = loss_patched.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss_patched * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs):
        latent, mask = self.forward_encoder(imgs)
        pred = self.decoder(latent)  # [N, L, p*p*3]
        return pred, mask

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        pred, mask = self.forward(imgs)
        loss = self.forward_loss(imgs, pred, mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        pred, mask = self.forward(imgs)
        loss = self.forward_loss(imgs, pred, mask)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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


def mae_swin_t(**kwargs):
    pretrained_weights = kwargs.pop('pretrain_path', None)
    model = MaskedAutoencoderSwin(
        patch_size=4, in_chans=3,  # swin defaults: patch size: 4
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],  # swin defaults: embed_dim=96
        mlp_ratio=4.0, **kwargs)

    if pretrained_weights:
        msg = model.load_state_dict(torch.load(pretrained_weights, map_location=model.device)['model'], strict=False)
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
