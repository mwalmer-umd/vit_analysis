"""
###########################################################################
Model wrapper for BEiT ViTs to extract both attention and features.

Based partly on code from:
https://github.com/microsoft/unilm/blob/master/beit

Written by: Saksham Suri
###########################################################################
"""
import sys

import torch
from torchvision import transforms as pth_transforms

from meta_utils.feature_extractor import FeatureExtractor
from meta_utils.block_mapper import block_mapper
from meta_utils.preproc import standard_transform
from timm.models import create_model

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

class BEIT_Wrapper:
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
            print('ERROR: Required imsize for BEIT is 224')
            exit(-1)
        self.mod_id = 'BEIT-ViT-%s-%i-%i'%(arch, patch, imsize)
        if self.mod_id == 'BEIT-ViT-B-16-224':
            self.checkpoint_file = 'models/beit/beit_base_patch16_224_pt22k.pth'
            self.arch = 'beit_base_patch16_224'
            self.num_layers = 12
            self.init_val = 0.1
        elif self.mod_id == 'BEIT-ViT-L-16-224':
            self.checkpoint_file = 'models/beit/beit_large_patch16_224_pt22k.pth'
            self.arch = 'beit_large_patch16_224'
            self.num_layers = 24
            self.init_val = 1e-5
        else:
            print('ERROR: Invalid BEIT config')
            exit(-1)
        # prepare transform
        self.transform = standard_transform('beit', imsize)
        # handle block selection
        self.blk_sel = blk_sel
        self.blk_idxs = block_mapper(arch, blk_sel)



    def load(self):
        # build model
        model = create_model(
            self.arch,
            pretrained=False,
            num_classes=0, #set non zero ?
            drop_rate=0,
            drop_path_rate=0,
            attn_drop_rate=0,
            drop_block_rate=None,
            # use_mean_pooling=False, #True for classification
            head_init_scale=0.001,
            # use_rel_pos_bias=True,
            # use_abs_pos_emb=False,
            # init_values=0.1, #1e-5 for large
        )
        patch_size = self.patch
        window_size = (self.imsize // patch_size, self.imsize // patch_size)
        patch_num = window_size[0] * window_size[1]
        checkpoint = torch.load(self.checkpoint_file, map_location='cpu')
        checkpoint_model = None
        model_keys = 'model|module'
        for model_key in model_keys.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        if "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
            print("Expand the shared relative position embedding to each transformer block. ")
            num_layers = self.num_layers
            rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
            for i in range(num_layers):
                checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()

            checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")
        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

            if "relative_position_bias_table" in key:
                rel_pos_bias = checkpoint_model[key]
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = model.state_dict()[key].size()
                dst_patch_shape = window_size
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()
                num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
                dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
                if src_size != dst_size:
                    print("Position interpolate for %s from %dx%d to %dx%d" % (
                        key, src_size, src_size, dst_size, dst_size))
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    print("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(num_attn_heads):
                        z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                        f = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                    rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                    new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                    checkpoint_model[key] = new_rel_pos_bias
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = patch_num
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        # load weights
        load_state_dict(model, checkpoint_model, prefix="")
        self.model = model

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
        acts = self.extractor(x.to(self.device))
        return acts