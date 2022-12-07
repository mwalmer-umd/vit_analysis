"""
###########################################################################
Model wrapper for CLIP ViTs to extract both attention and features.

Utilizes code from the original CLIP repository:
https://github.com/openai/CLIP

Written by: Matthew Walmer
###########################################################################
"""
import sys
import torch
import clip

from meta_utils.feature_extractor import FeatureExtractor
from meta_utils.block_mapper import block_mapper
from meta_utils.preproc import standard_transform


class CLIP_Wrapper:
    def __init__(self, arch, patch, imsize, extract_mode='none', blk_sel='all'):
        assert extract_mode in ['none', 'attn', 'feat']
        if extract_mode == 'none':
            print('WARNING: wrapper running in NONE mode, no tensors will be extracted')
            print('only use this mode if extracting features separately')
        self.arch = arch
        self.patch = patch
        self.imsize = imsize
        self.extract_mode = extract_mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # create model identifier and test configuration
        self.clip_arch = 'ViT-%s/%i'%(arch, patch)
        if imsize != 224:
            self.clip_arch += '@%ipx'%imsize
        self.patch_size = patch
        valid_models = clip.available_models()
        if self.clip_arch not in valid_models:
            print('ERROR: invalid clip config: ' + self.clip_arch)
            print('valid options:')
            print(clip.available_models())
            print('(default imsize 224)')
            exit(-1)
        self.mod_id = ('CLIP-%s'%self.clip_arch).replace('/','-')
        if '@' in self.mod_id:
            self.mod_id = self.mod_id.replace('@','-').replace('px','')
        else:
            self.mod_id = self.mod_id + '-%i'%imsize
        # transform
        self.transform = standard_transform('clip', imsize)
        # handle block selection
        self.blk_sel = blk_sel
        self.blk_idxs = block_mapper(arch, blk_sel)



    def load(self):
        self.model, self.orig_transform = clip.load(self.clip_arch, device=self.device)
        self.model.eval()
        # prepare hooks - depending on extract_mode
        layers = []
        for idx in self.blk_idxs:
            if self.extract_mode == 'none':
                continue
            if self.extract_mode == 'attn':
                layers.append(self.model.visual.transformer.resblocks[idx].attn_holder)
            if self.extract_mode == 'feat':
                layers.append(self.model.visual.transformer.resblocks[idx])
        self.extractor = FeatureExtractor(self.model, layers, runner=self.model.encode_image)



    def get_activations(self, x):
        acts = self.extractor(x.to(self.device))
        if self.extract_mode == 'feat': # fix dimension order for features
            fixed_acts = []
            for a in acts:
                a = torch.moveaxis(a, 1, 0)
                fixed_acts.append(a)
            return fixed_acts
        return acts