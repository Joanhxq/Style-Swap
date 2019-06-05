# -*- coding: utf-8 -*-

from torch import nn
import torch.nn.functional as F
import torch

def style_swap(cf, sf, opt):  # cf,sf  Batch_size x C x H x W
    b, c, h, w = sf.size()  # 2 x 256 x 64 x 64

    # Create convolutional filters by style features    
    h_t, w_t = h // opt.patch_size, w // opt.patch_size
    sf = sf[:, :, :h_t*opt.patch_size, :w_t*opt.patch_size]
    sf_chunks = torch.chunk(sf, opt.patch_size, dim=0)    # separate different style feature maps
    sf_patches = torch.Tensor([]).to(device=opt.device)    # store style patches
    for _, sf_chunk in enumerate(sf_chunks):
        sf_split = torch.split(sf_chunk, opt.patch_size, dim=2)
        for _, sf_s in enumerate(sf_split):
            sf_split_t = torch.split(sf_s, opt.patch_size, dim=3)
            for _,sf_t in enumerate(sf_split_t):
                sf_patches = torch.cat((sf_patches, sf_t), dim=0)
    # patches size is (441*2) x 256 x 3 x 3
    
    sf_patch = torch.chunk(sf_patches, opt.minibatch, dim=0)   # separate different style patches
    transconv = torch.Tensor([]).to(device=opt.device)
    for _, sf_p in enumerate(sf_patch):
        sp_norm = sf_p / torch.norm(sf_p, dim=1).unsqueeze(1)
        weight_norm = nn.Parameter(data=sp_norm, requires_grad=False)  # size: 441 x 256 x 3 x 3
        conv = F.conv2d(cf, weight_norm, stride=1)    # output size: 2 x 441 x 62 x 62
    
        max_value = torch.max(conv, 1)[0].unsqueeze(1)
        K_bar_zeros = torch.zeros_like(conv)
        K_bar_ones = torch.ones_like(conv)
        K_bar = torch.where(torch.ge(conv, max_value), K_bar_ones, K_bar_zeros)
    
        weight = nn.Parameter(data=sf_p, requires_grad=False)
        transconv_t = F.conv_transpose2d(K_bar, weight, stride=1)  # output size: 2 x 256 x 64 x 64 -> \Phi^ss(C,S)
        transconv = torch.cat((transconv, transconv_t), dim=0)
    return transconv / 9.0   # size : 4 x 256 x 64 x 64
        
