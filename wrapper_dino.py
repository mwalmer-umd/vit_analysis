"""
###########################################################################
Model wrapper for DINO ViTs to extract both attention and features.

Utilizes code from the original DINO repository:
(https://github.com/facebookresearch/dino) Original copyright notice below.

Written by: Matthew Walmer
###########################################################################
"""
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os

import torch
from torchvision import transforms as pth_transforms

from meta_utils.arch_conv import letter2arch
from meta_utils.feature_extractor import FeatureExtractor
from meta_utils.block_mapper import block_mapper
from meta_utils.preproc import standard_transform
sys.path.append('dino/')
import vision_transformer as vits



class DINO_Wrapper:
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
        self.mod_id = 'DINO-ViT-%s-%i-%i'%(arch, patch, imsize)
        if arch not in ['S','B'] or patch not in [8, 16]:
            print('ERROR: Invalid DINO config ' + self.mod_id)
            exit(-1)
        # transform
        self.transform = standard_transform('dino', imsize)
        # handle block selection
        self.blk_sel = blk_sel
        self.blk_idxs = block_mapper(arch, blk_sel)
    

    def load(self):
        # load model
        dino_arch = 'vit_%s'%letter2arch(self.arch)
        self.model = vits.__dict__[dino_arch](patch_size=self.patch, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.to(self.device)
        url = None
        if dino_arch == "vit_small" and self.patch == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif dino_arch == "vit_small" and self.patch == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif dino_arch == "vit_base" and self.patch == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif dino_arch == "vit_base" and self.patch == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)
        else:
            print("ERROR: There is no reference weights available for this model")
            exit(-1)
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