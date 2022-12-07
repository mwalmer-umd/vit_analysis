"""
###########################################################################
Model wrapper for timm ViTs (fully supervised) to extract both attention
and features.

Written by: Saksham Suri
###########################################################################
"""
import torch
import timm

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from meta_utils.arch_conv import letter2arch
from meta_utils.feature_extractor import FeatureExtractor
from meta_utils.block_mapper import block_mapper
from meta_utils.preproc import standard_transform



class TIMM_Wrapper:
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
        self.timm_arch = 'vit_%s_patch%i_%i'%(letter2arch(arch), patch, imsize)
        valid_timm_archs = ['vit_base_patch8_224', 'vit_base_patch16_224', 'vit_base_patch32_224', 'vit_small_patch16_224' , \
                            'vit_small_patch32_224', 'vit_large_patch16_224']
        if self.timm_arch not in valid_timm_archs:
            print('ERROR: Invalid timm arch')
            print('valid options:')
            print(valid_timm_archs)
            exit(-1)
        self.mod_id = 'TIMM-ViT-%s-%i-%i'%(arch, patch, imsize)
        # transform
        self.transform = standard_transform('timm', imsize)
        # handle block selection
        self.blk_sel = blk_sel
        self.blk_idxs = block_mapper(arch, blk_sel)



    def load(self):
        # load model
        self.model = timm.create_model(self.timm_arch, pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        # prepare transform        
        self.config = resolve_data_config({}, model=self.model)
        self.orig_transform = create_transform(**self.config)
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
        acts = self.extractor(x.to(self.device))
        return acts