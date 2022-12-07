# helper to extract dense features for local tasks like segmentation
# some models do not support variable input size, so this wrapper will
# automatically split an input image to 224 x 224 patches and compute and
# combine features from all the regions.
import math

import torch
import PIL
import numpy as np

from meta_utils.preproc import minimal_transform

# automatically identifies the meta model and patch size from the mod_id
# if cpu_assembly == True, the features will be gathered on CPU instead of GPU to save memory
class dense_extractor():
    def __init__(self, mod_wrap, window=224, batch_limit=4, tile_factor=2, cpu_assembly=False):
        self.mod_wrap = mod_wrap
        if self.mod_wrap.extract_mode != 'feat':
            print('ERROR: dense_extractor can only be run with a model wrapper in feat mode')
            exit(-1)
        self.window = window
        self.batch_limit = batch_limit
        self.tile_factor = tile_factor
        self.cpu_assembly = cpu_assembly
        self.mod_id = self.mod_wrap.mod_id + '-dense'
        self.device = self.mod_wrap.device
        self.transform = None
        meta_model = self.mod_wrap.mod_id.split('-')[0].lower()
        self.post_transform = minimal_transform(meta_model)
        self.patch = int(self.mod_wrap.mod_id.split('-')[3])
        if not cpu_assembly:
            self.assembly_device = self.mod_wrap.device
        else:
            self.assembly_device = "cpu"


    def load(self):
        self.mod_wrap.load()


    def get_resize_dim(self, x):
        orig_w, orig_h = x.size
        if orig_w < orig_h:
            w = int(self.window * self.tile_factor)
            h = (w / orig_w) * orig_h
            h = int((h // self.patch)*self.patch)
        else:
            h = int(self.window * self.tile_factor)
            w = (h / orig_h) * orig_w
            w = int((w // self.patch)*self.patch)
        return [w, h]


    def slice_image(self, x):
        # resize
        res_dim = self.get_resize_dim(x)
        x_res = x.resize(res_dim, PIL.Image.Resampling.BICUBIC)
        x_res = np.array(x_res)
        # gather slices
        slices = []
        start_coords = []
        nv = math.ceil(x_res.shape[0] / self.window)
        nh = math.ceil(x_res.shape[1] / self.window)
        for i_v in range(nv):
            for i_h in range(nh):
                start_v = min(i_v*self.window, max(x_res.shape[0]-self.window, 0))
                start_h = min(i_h*self.window, max(x_res.shape[1]-self.window, 0))
                coords = [start_v, start_h]
                start_coords.append(coords)
                cur_slice = x_res[start_v:start_v+self.window, start_h:start_h+self.window]
                slices.append(cur_slice)
        return slices, start_coords, x_res


    # do not apply any pre-processing to x, x should be the directly loaded PIL Image
    def get_activations(self, x):
        slices, start_coords, x_res = self.slice_image(x)
        acts = None # for gathering the activations
        mask = None # for masking filled regions
        batch = []
        batch_sc = []
        for i in range(len(slices)):
            # gather batch
            batch.append(self.post_transform(slices[i]).to(self.device))
            batch_sc.append(start_coords[i])
            if len(batch) < self.batch_limit and i < (len(slices)-1):
                continue
            # prep images
            x = torch.stack(batch, dim=0)
            # get features
            f = self.mod_wrap.get_activations(x)
            if self.cpu_assembly:
                f_cpu = []
                for f_gpu in f:
                    f_cpu.append(f_gpu.cpu())
                f = f_cpu
            f = torch.stack(f, 0)
            # prep activation gathering array
            if acts is None:
                # get number of tokens and grid shape
                nb = f.shape[0]
                nt = f.shape[2]
                nf = f.shape[3]
                gs = int(math.sqrt(nt-1)) # tokens along slice edge
                if (gs*gs)+1 != nt:
                    print('ERROR: dense_extractor requires a square token array')
                    exit(-1)
                ps = int(self.window / gs) # model patch size
                # prep activation gathering array
                gv = int(x_res.shape[0] / ps) # vertical tokens in overall image
                gh = int(x_res.shape[1] / ps) # horizontal tokens in overall image
                acts = torch.zeros([nb, gv, gh, nf], dtype=f.dtype, device=self.assembly_device)
                mask = torch.ones([1, gv, gh, 1], dtype=f.dtype, device=self.assembly_device)
            # overlay features
            ns = f.shape[1]
            for j in range(ns):
                f_res = f[:,j,1:,:] # remove CLS token
                f_res = f_res.reshape([nb, gs, gs, nf])
                s0 = int(batch_sc[j][0] / ps)
                s1 = int(batch_sc[j][1] / ps)
                m = mask[:, s0:s0+gs, s1:s1+gs, :]
                mf = m * f_res
                acts[:, s0:s0+gs, s1:s1+gs, :] += mf
                mask[:, s0:s0+gs, s1:s1+gs, :] = 0
            # reset batch
            batch = []
            batch_sc = []
        return acts