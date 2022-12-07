"""
###########################################################################
A dummy wrapper that only returns random Gaussian noise for all attention
and feature outputs. Returns outputs in the same shape and format of a
ViT-B/16 model with input size 224x224. Used to compute random-chance
scores for the various metrics.

Written by: Matthew Walmer
###########################################################################
"""
import sys
import os

import torch
import numpy as np

from meta_utils.arch_conv import letter2arch
from meta_utils.feature_extractor import FeatureExtractor
from meta_utils.block_mapper import block_mapper
from meta_utils.preproc import standard_transform
sys.path.append('dino/')
import vision_transformer as vits



class Random_Wrapper:
    def __init__(self, arch, patch, imsize, extract_mode='none', blk_sel='all'):
        assert extract_mode in ['none', 'attn', 'feat']
        if extract_mode == 'none':
            print('WARNING: Random wrapper should not be run with extract_mode=none, as it will do nothing at all')
            # exit(-1)
        self.arch = arch
        self.patch = patch
        self.imsize = imsize
        self.extract_mode = extract_mode        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # create model identifier and test configuration
        self.mod_id = 'RANDOM-ViT-%s-%i-%i'%(arch, patch, imsize)
        if self.mod_id != 'RANDOM-ViT-B-16-224':
            print('ERROR: Random wrapper can only run in a ViT-B-16-224 configuration')
            exit(-1)
        self.nb = 12
        self.nh = 12
        self.nt = 197
        self.nf = 768
        # transform
        self.transform = standard_transform('random', imsize) # kept as placeholder
        # handle block selection
        self.blk_sel = blk_sel
        self.blk_idxs = block_mapper(arch, blk_sel)
        if blk_sel != 'all':
            print('WARNING: Random wrapper can only run with blk_sel=all')
            print('Will return a tensor of size [bs, nf, nt] in this case')
        #     exit(-1)

    def load(self):
        return


    def get_activations(self, x):
        bs = x.shape[0]
        if self.extract_mode == 'attn':
            r = torch.rand(size=[self.nb, bs, self.nh, self.nt, self.nt], dtype=torch.float32, device=self.device)
            m = torch.mean(r, dim=3, keepdim=True)
            r = r/m
        else: # feat
            if(self.blk_sel == 'all'):
                # r = torch.rand(size=[self.nb, bs, self.nf, self.nt], dtype=torch.float32, device=self.device)
                r = torch.rand(size=[self.nb, bs, self.nt, self.nf], dtype=torch.float32, device=self.device)
            else:
                return [torch.rand(size=[bs, self.nt, self.nf], dtype=torch.float32, device=self.device)]
        acts = []
        for i in range(self.nb):
            acts.append(r[i,...])
        return acts