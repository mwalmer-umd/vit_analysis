"""
###########################################################################
Model wrapper for MAE ViTs to extract both attention and features.

Based partly on mea/demo/mae_visualize.ipynb from the original MAE repo:
https://github.com/facebookresearch/mae

Written by: Matthew Walmer
###########################################################################
"""
import sys

import torch
from torchvision import transforms as pth_transforms

from meta_utils.feature_extractor import FeatureExtractor
from meta_utils.block_mapper import block_mapper
from meta_utils.preproc import standard_transform
sys.path.append('mae/')
import models_mae



class MAE_Wrapper:
    def __init__(self, arch, patch, imsize, extract_mode='none', blk_sel='all'):
        assert extract_mode in ['none', 'attn', 'feat']
        if extract_mode == 'none':
            print('WARNING: wrapper running in NONE mode, no tensors will be extracted')
            print('only use this mode if extracting features separately')
        self.arch = arch
        self.patch = patch
        self.imsize = imsize
        self.extract_mode = extract_mode
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # create model identifier and test configuration
        if imsize != 224:
            print('ERROR: Required imsize for MAE is 224')
            exit(-1)
        self.mod_id = 'MAE-ViT-%s-%i-%i'%(arch, patch, imsize)
        if self.mod_id == 'MAE-ViT-B-16-224':
            self.checkpoint_file = 'models/mae/mae_pretrain_vit_base.pth'
            self.arch = 'mae_vit_base_patch16'
        elif self.mod_id == 'MAE-ViT-L-16-224':
            self.checkpoint_file = 'models/mae/mae_pretrain_vit_large.pth'
            self.arch = 'mae_vit_large_patch16'
        elif self.mod_id == 'MAE-ViT-H-14-224':
            self.checkpoint_file = 'models/mae/mae_pretrain_vit_huge.pth'
            self.arch = 'mae_vit_huge_patch14'
        else:
            print('ERROR: Invalid MAE config')
            exit(-1)
        # prepare transform
        self.transform = standard_transform('mae', imsize)
        # handle block selection
        self.blk_sel = blk_sel
        self.blk_idxs = block_mapper(arch, blk_sel)


    def load(self):
        # build model
        self.model = getattr(models_mae, self.arch)()
        # load model
        checkpoint = torch.load(self.checkpoint_file, map_location='cpu')
        msg = self.model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        self.model.eval()
        self.model.to(self.device)
        # prepare hooks - depending on extract_mode
        layers = []
        for idx in self.blk_idxs:
            if self.extract_mode == 'none':
                continue
            if self.extract_mode == 'attn':
                layers.append(self.model.blocks[idx].attn.attn_drop)
            if self.extract_mode == 'feat':
                layers.append(self.model.blocks[idx])
        self.extractor = FeatureExtractor(self.model, layers)


    def get_activations(self, x):
        acts = self.extractor(x.to(self.device), mask_ratio=0.0, no_shuffle=True)
        return acts