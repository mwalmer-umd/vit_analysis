import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models

import vits

model_names = ['vit_small', 'vit_base']

def load_model(arch='vit_base_patch16_224'):
    if arch=='vit_base_patch16_224':
        arch = 'vit_base'
        pretrained = 'moco-v3/weights/vit-b-300ep.pth.tar'
    elif arch=='vit_small_patch16_224':
        arch = 'vit_small'
        pretrained = 'moco-v3/weights/vit-s-300ep.pth.tar'
    print("=> creating model '{}'".format(arch))
    model = vits.__dict__[arch]()
    linear_keyword = 'head'
    for name, param in model.named_parameters():
        param.requires_grad = False
    

    # load from pre-trained, before DistributedDataParallel constructor
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

        print("=> loaded pre-trained model '{}'".format(pretrained))
   
    model = model.cuda()
    model.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    return model, transform


if __name__ == '__main__':
    main()
