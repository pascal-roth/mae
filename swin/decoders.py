# File for different encoder models

from json import encoder
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# import scripts
from self_sup_seg.third_party.mae.swin.utils import Up, UpConv


class DecoderFPN(torch.nn.Module):
    """
    Feature Pyramide Network
    Hierarchical decoder implementation where the last encoder is upscaled with convolutional layers. 
    In addition, the different stages of the encoder are given as additional input to the corresponding decoder layers.
    """

    def __init__(self, 
                 out_indicies: List[int],
                 num_layers: int,
                 embed_dim: int,
                ) -> None:
        super().__init__()

        self.out_indicies: List[int] = out_indicies
        self.encoder_embed_dim: int = embed_dim
        self.num_layers: int = num_layers

        # build model
        self.blocks = nn.ModuleList([])

        ## dimension is downscaled by a factor of 2 in each Swin Block
        in_dim = self.encoder_embed_dim * (2 ** (self.num_layers-1))
        out_dim = in_dim

        ## build layers for every stage
        first_layer = True
        for i in range(self.num_layers-1, 0, -1):            
            out_dim /= 2
            
            if i not in self.out_indicies:
                continue                

            if first_layer:
                ## first layer only get the encoder last layer as input
                self.blocks.append(UpConv(int(in_dim), int(out_dim)))
                first_layer = False
            else:
                ## in_dim doubled since contactenated with encoder tensor of same channel size
                self.blocks.append(Up(int(in_dim*2), int(out_dim)))
            
            in_dim = out_dim

        # final prediction layer:
        self.blocks.append(torch.nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
            UpConv(self.encoder_embed_dim, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
            UpConv(128, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
            nn.Conv2d(256, 3, kernel_size=3, padding=1, bias=False),
        ))
    def forward(self, outs) -> torch.Tensor:
        x = self.blocks[0](outs[f'res{self.out_indicies[-1]+2}'])
        
        for idx, blk in enumerate(self.blocks[1:-1]):
            x = blk(x, outs[f'res{self.out_indicies[-idx-2]+2}'])

        return self.blocks[-1](x)
