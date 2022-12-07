"""
###########################################################################
Cluster Analysis for Segmentation Feature Purity

Written by: Matthew Walmer
###########################################################################
"""
import os
import argparse
import time
import shutil
import math

import numpy as np
import torch
from torchvision.datasets import CocoDetection
from PIL import Image

from meta_utils.get_model_wrapper import get_model_wrapper
from meta_utils.result_cacher import read_results_cache, save_results_cache
from meta_utils.dense_extractor import dense_extractor
from meta_utils.simple_progress import SimpleProgress
from cka.utils import cluster_metric
from analysis.attention_plots import block_level_plots, head_level_plots, meta_plot, meta_dataload
from meta_utils.data_summary import best_block_table
from partimagenet_dataset import PartImagenetDataset
from run_attention_analysis import select_positions



#################### PRE-PROCESSING ####################



'''
input:
-ann - annotations for the image
-dataset - the coco dataset object
-gv - vertical size of the token grid (int)
-gh - horizonatal size of the token grid (int)
-ccrop - center crop the token mask to a square image before resizing. This
        should be enabled depending on what pre-processing is used by the ViT

return: an array of "soft" masks of shape [N,KV,KH] where N = the
number of objects in the scene, and KV x KH is the size of the token grid.
Also returns a numpy array of size [N] with the corresponding class labels.
soft masks not binary, result of resizing the binary mask with bicubic
rescaling. They are rescale to sum to 1, for use with weighted averages of
feature maps. In addition, returns a binary mask for all annotated objects
combined.
'''
def prep_token_masks(ann, dataset, gv, gh, ccrop=False, silent=True):
    masks = []
    labs = []
    skips = 0
    loads = 0
    for a in ann:
        l = a['category_id']
        # check for erroneous masks in PartImagenet
        if a['iscrowd'] != 1:
            seg = a['segmentation'][0]
            if len(seg) <= 4:
                if not silent:
                    print('WARNING: found a segmentation anno with invalid length <=4, skipping')
                    print(a)
                skips += 1
                continue
        # get original mask
        m = dataset.coco.annToMask(a)
        m_orig = m
        # process mask
        if ccrop: # center cropping
            h, w = m.shape
            if h < w:
                ws = int((w-h)/2)
                m = m[:, ws:ws+h]
            else:
                hs = int((h-w)/2)
                m = m[hs:hs+w, :]
        m = Image.fromarray(m*255)
        m = m.resize([gh, gv])
        m = np.array(m).astype(float)
        if np.sum(m) == 0:
            skips += 1            
            continue
        m /= np.sum(m)
        masks.append(m)
        labs.append(l)
        loads += 1
    if loads == 0:
        return None, None, skips, loads, None
    masks = np.stack(masks, axis=0)
    labs = np.array(labs)
    # form merged binary mask of all classes
    bin_mask = np.sum(masks, axis=0)
    bin_mask = (bin_mask > 0).astype(float)
    return masks, labs, skips, loads, bin_mask



#################### POST-PROCESSING ####################


'''
Convert attention maps into binary attention maps in one
of several modes:
-cls - use the cls token attention map
-spc - take the average of all spatial attention maps
-center - use the center-most spatial token

mass determines what fraction of the total attention
mass (after removing cls position) is kept when creating
the binary mask
'''
def att_to_binary_mask(attn, mode='cls', merge=False, mass=0.6):
    assert mode in ['cls', 'cls', 'spc', 'center']
    nb = attn.shape[0]
    nh = attn.shape[1]
    nt = attn.shape[2]
    gl = int(math.sqrt(nt-1))
    # mask mode
    attn = attn[:,:,:,1:] # remove CLS token from destination tokens
    if mode == 'cls': 
        attn = attn[:,:,0,:] # keep CLS token
    elif mode == 'spc':
        attn = attn[:,:,1:,:] # remove CLS token
        attn = torch.sum(attn, dim=2) # average all spatial positions
    else:
        positions, _ = select_positions(gl, gl)
        c = positions[3]
        attn = attn[:,:,c,:] # keep central spatial token
    # merge heads of each block
    if merge:
        attn = torch.sum(attn, dim=1, keepdim=True)
        nh = 1
    # normalize for the removed mass
    attn /= torch.sum(attn, dim=2, keepdim=True) 
    # make mask (based on code from DINO's visualize_attention.py)
    val, idx = torch.sort(attn)
    val /= torch.sum(val, dim=2, keepdim=True)
    cumval = torch.cumsum(val, dim=2)
    th_attn = cumval > (1 - mass)
    idx2 = torch.argsort(idx)
    for b in range(nb):
        for h in range(nh):
            th_attn[b,h] = th_attn[b,h][idx2[b,h]]
    th_attn = th_attn.reshape(nb, nh, gl, gl).float()
    return th_attn    



#################### MASK ALIGNMENT METRICS ####################



def iou_score(m_pred, m_gt):
    m_gt = torch.unsqueeze(m_gt, 0)
    m_gt = torch.unsqueeze(m_gt, 0)
    s = m_pred + m_gt
    i = torch.sum(s > 1, dim=(2,3))
    u = torch.sum(s > 0, dim=(2,3))
    iou = i / u
    return iou



#################### ANALYSIS ####################



def prep_dataset(args):
    if args.dataset == 'coco':
        root = os.path.join(args.cocoroot, 'images', args.cocopart)
        anno = os.path.join(args.cocoroot, 'annotations', 'instances_%s.json'%args.cocopart)
        dataset = CocoDetection(root=root, annFile=anno)
        prog_end = args.imcount
    elif args.dataset == 'pin':
        dataset = PartImagenetDataset(args.pinroot, args.persc)
        prog_end = len(dataset)
    else:
        print('INVALID --dataset choice')
        exit(-1)
    return dataset, prog_end



# extract and cache image features and labels 
def extract_feats(args): 
    # prep model wrapper
    mod_wrap = get_model_wrapper(args.meta_model, args.arch, args.patch, args.imsize, 'feat', 'all')
    if args.dense:
        mod_wrap = dense_extractor(mod_wrap, batch_limit=args.batch, cpu_assembly=args.cpu_assembly)
    print(mod_wrap.mod_id)
    # check for existing feats
    cache_dir = os.path.join(args.output_dir, '_cache', mod_wrap.mod_id)
    cache_feat_file = os.path.join(cache_dir, '%s_features.npy'%args.dataset)
    cache_lab_file = os.path.join(cache_dir, '%s_labels.npy'%args.dataset)
    if os.path.isfile(cache_feat_file) and os.path.isfile(cache_lab_file):
        print('found existing cached features')
        if args.overcache:
            print('--overcache enabled: cached features will be over-written')
        else:
            print('continuing...')
            return
    # load model
    mod_wrap.load()
    # prep dataset
    dataset, prog_end = prep_dataset(args)
    # gathering features
    print('Gathering Features...')
    all_feats = []
    all_labs = []
    t0 = time.time()
    anno_skips = 0
    anno_loads = 0
    imc = 0
    SP = SimpleProgress(end=prog_end, step=1)
    for img, ann in dataset:
        SP.update()
        # limit image loading
        imc += 1
        if args.dataset == 'coco' and imc >= args.imcount: break
        # empty annotations
        if len(ann) == 0: continue
        # run image
        if not args.dense:
            x = mod_wrap.transform(img)
            x = torch.unsqueeze(x, 0).to(mod_wrap.device)
        else:
            x = img
        fs = mod_wrap.get_activations(x)
        # post-processing
        if not args.dense:
            fs = torch.cat(fs)
            fs = fs[:,1:,:] # remove cls token
            nb = fs.shape[0]
            nt = fs.shape[1]
            nf = fs.shape[2]
            gs = int(math.sqrt(nt))
            fs = fs.reshape([nb, gs, gs, nf])
        fs = fs.unsqueeze(0) # expand dim for N masks
        # process masks
        gv = fs.shape[2] # token grid vertical size
        gh = fs.shape[3] # token grid vertical size
        masks, labs, skips, loads, _ = prep_token_masks(ann, dataset, gv, gh, ccrop=(not args.dense))
        if loads == 0: continue # no annotations loaded
        masks = torch.Tensor(masks)
        if not (args.dense and args.cpu_assembly):
            masks = masks.to(mod_wrap.device)
        masks = masks.unsqueeze(3)
        masks = masks.unsqueeze(1)
        anno_skips += skips
        anno_loads += loads
        # gather features
        feats = masks * fs
        feats = torch.sum(feats, dim=(2,3))
        all_feats.append(feats.cpu().numpy())
        all_labs.append(labs)
    SP.finish() 
    print('%i gt masks were included'%anno_loads)
    print('%i gt masks were skipped (too small or cut off)'%anno_skips)
    # save files
    print('caching features...')
    all_feats = np.concatenate(all_feats, axis=0)
    all_labs = np.concatenate(all_labs, axis=0)
    print(all_feats.shape)
    print(all_labs.shape)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_feat_file, all_feats)
    np.save(cache_lab_file, all_labs)



def load_or_run_analysis(args):
    # prep model wrapper
    mod_wrap = get_model_wrapper(args.meta_model, args.arch, args.patch, args.imsize, 'feat', 'all')
    if args.dense:
        mod_wrap = dense_extractor(mod_wrap, batch_limit=args.batch, cpu_assembly=args.cpu_assembly)
    # check cache
    analysis_methods = ['seg-feat-nmi', 'seg-feat-ari', 'seg-feat-pur']
    for i in range(len(analysis_methods)):
        analysis_methods[i] += '_[%s]'%args.dataset
    if not (args.nocache or args.overcache):
        results, found, not_found = read_results_cache(mod_wrap.mod_id, analysis_methods)
        if len(results) == 3:
            return results, analysis_methods
    # load cached feats
    print('loading cached features...')
    cache_dir = os.path.join(args.output_dir, '_cache', mod_wrap.mod_id)
    cache_feat_file = os.path.join(cache_dir, '%s_features.npy'%args.dataset)
    cache_lab_file = os.path.join(cache_dir, '%s_labels.npy'%args.dataset)
    if not (os.path.isfile(cache_feat_file) and os.path.isfile(cache_lab_file)):
        print('ERROR: could not find cached features for model: %s'%mod_wrap.mod_id)
        print('  %s'%cache_feat_file)
        print('  %s'%cache_lab_file)
        print('--run_feats must be run before using --run_met')
        exit(-1)
    all_feats = np.load(cache_feat_file)
    all_labs = np.load(cache_lab_file)
    # run clustering
    nb = all_feats.shape[1]
    t0 = time.time()
    print('Gathered Features of Size:')
    print(all_feats.shape)
    print('Running clustering...')
    all_nmi = []
    all_ari = []
    all_pur = []
    for i in range(nb):
        print('Block %i'%i)
        nmi, ari, pur = cluster_metric(80, all_feats[:,i,:], all_labs)
        all_nmi.append(nmi)
        all_ari.append(ari)
        all_pur.append(pur)
    all_nmi = np.array(all_nmi)
    all_ari = np.array(all_ari)
    all_pur = np.array(all_pur)
    results = [all_nmi, all_ari, all_pur]
    print('done in %.2f minutes'%((time.time()-t0)/60))
    # cache results
    if not args.nocache:
        save_results_cache(results, mod_wrap.mod_id, analysis_methods)
    return results, analysis_methods



# metrics that test how well the attention maps align with ground
# truth semantic segmentation masks, to identify semantic-focused heads
def load_or_run_att_align_analysis(args):
    assert not args.dense
    # prep model wrapper
    mod_wrap = get_model_wrapper(args.meta_model, args.arch, args.patch, args.imsize, 'attn', 'all')
    # prepare masking modes
    mask_methods = ['cls', 'spc', 'center']
    analysis_methods = []
    analysis_configs = []
    for mm in mask_methods:
        analysis_methods.append('%s_att_align_iou'%mm)
        analysis_methods.append('%s-merge_att_align_iou'%mm)
        analysis_configs.append((mm, False))
        analysis_configs.append((mm, True))
    for i in range(len(analysis_methods)): # add dataset specifier
        analysis_methods[i] += '_[%s]'%args.dataset
    all_dict = {} # gather results here
    for a_m in analysis_methods:
        all_dict[a_m] = []
    # check cache
    if not (args.nocache or args.overcache):
        results, found, not_found = read_results_cache(mod_wrap.mod_id, analysis_methods)
        if len(results) == len(analysis_methods):
            return results, analysis_methods
    # load model
    mod_wrap.load()
    # prep dataset
    dataset, prog_end = prep_dataset(args)
    # gathering features
    print('Running Attention Alignment Metrics...')
    anno_skips = 0
    anno_loads = 0
    imc = 0
    SP = SimpleProgress(end=prog_end, step=1)
    for img, ann in dataset:
        SP.update()
        # limit image loading for coco
        imc += 1
        if args.dataset == 'coco' and imc > args.imcount: break
        # empty annotations
        if len(ann) == 0: continue
        # run image
        x = mod_wrap.transform(img)
        x = torch.unsqueeze(x, 0).to(mod_wrap.device)
        fs = mod_wrap.get_activations(x)
        fs = torch.cat(fs)
        # process gt masks
        nt = fs.shape[2]
        gl = int(math.sqrt(nt-1)) # grid edge length
        _, _, skips, loads, bin_mask = prep_token_masks(ann, dataset, gl, gl, ccrop=True)
        if loads == 0: continue # no annotations loaded (due to cropping or too small)
        bin_mask = torch.Tensor(bin_mask).to(mod_wrap.device)
        anno_skips += skips
        anno_loads += loads
        # generate masks and compute ious
        for am_idx in range(len(analysis_methods)):
            a_m = analysis_methods[am_idx]
            mask_type, merge_type = analysis_configs[am_idx]
            att_mask = att_to_binary_mask(fs, mask_type, merge_type, args.mass)
            att_iou = iou_score(att_mask, bin_mask)
            if merge_type:
                att_iou = att_iou[:,0] # remove extra dim
            all_dict[a_m].append(att_iou.cpu().numpy())
    SP.finish() 
    # stack results
    results = []
    for a_m in analysis_methods:
        res = all_dict[a_m]
        res = np.stack(res, axis=0)
        results.append(res)
    # cache results
    if not args.nocache:
        save_results_cache(results, mod_wrap.mod_id, analysis_methods)
    return results, analysis_methods



#################### RUNNING MODES ####################



def run_metrics(args):
    mod_wrap = get_model_wrapper(args.meta_model, args.arch, args.patch, args.imsize, 'feat', 'all')
    mod_id = mod_wrap.mod_id
    if args.dense:
        mod_id += '-dense'
    print(mod_id)
    results, analysis_methods = load_or_run_analysis(args)
    # make plots
    print('making plots...')
    output_dir = os.path.join(args.output_dir, mod_id)
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(analysis_methods)):
        a_m = analysis_methods[i]
        res = results[i]
        block_level_plots(output_dir, mod_id, res, a_m)



def run_iou_met(args):
    if args.dense:
        print('WARNING: Attention alignment metrics cannot be run with --dense')
        return
    results, analysis_methods = load_or_run_att_align_analysis(args)
    # make plots
    print('making plots...')
    mod_wrap = get_model_wrapper(args.meta_model, args.arch, args.patch, args.imsize, 'attn', 'all')
    mod_id = mod_wrap.mod_id
    print(mod_id)
    output_dir = os.path.join(args.output_dir, mod_id)
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(analysis_methods)):
        a_m = analysis_methods[i]
        res = results[i]
        print(a_m)
        if 'merge' in a_m:
            block_level_plots(output_dir, mod_id, res, a_m)
        else:
            head_level_plots(output_dir, mod_id, res, a_m)



def run_meta(args):
    cache_dir = 'all_results'
    dirs = os.listdir(cache_dir)
    if len(dirs) == 0:
        print('WARNING: can only run meta analysis on cached results. no cached results found')
        return
    dirs.sort()
    print('Found %i cached results'%len(dirs))
    print(dirs)
    meta_out_dir = os.path.join(args.output_dir, '_meta')
    os.makedirs(meta_out_dir, exist_ok=True)

    # === Cluster Purity Analysis ===
    meta_plot_metrics = ['seg-feat-nmi_[coco]', 'seg-feat-ari_[coco]', 'seg-feat-pur_[coco]', 'seg-feat-nmi_[pin]', 'seg-feat-ari_[pin]', 'seg-feat-pur_[pin]']
    meta_plot_types = ['block_p', 'block_p', 'block_p', 'block_p', 'block_p', 'block_p']
    # meta_plot_metrics = ['seg-feat-pur_[coco]', 'seg-feat-pur_[pin]']
    # meta_plot_types = ['block_p', 'block_p']
    # combined meta plots:
    for i in range(len(meta_plot_metrics)):
        print(meta_plot_metrics[i])

        # OLD: all models, no pooling
        # meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pooled=False, inc_filters='dense')

        # OLD: all models, with pooling by type
        # meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pooled=True, inc_filters='dense')
        
        # B-16 models only
        # meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pooled=False, inc_filters=['dense', 'B-16'], suff='B-16', base_colors=True, x_block=True, separate_legend=True, no_arch=True)

        # all models, pertype style plot
        # meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pertype=True, inc_filters=['dense'])

        # B-16 models only (not dense)
        meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pooled=False, inc_filters=['B-16'], exc_filters=['dense'], suff='B-16-not-dense', base_colors=True, x_block=True, separate_legend=True, no_arch=True)

        # all models, pertype style plot (not dense)
        meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pertype=True, exc_filters=['dense'], suff='not-dense')

    # === IoU Analysis ===
    # mask_methods = ['cls', 'spc', 'center']
    datasets = ['pin', 'coco']
    mask_methods = ['cls', 'spc']
    # datasets = ['pin']
    meta_plot_metrics = []
    meta_plot_types = []
    for mm in mask_methods:
        for d in datasets:
            meta_plot_metrics.append('%s_att_align_iou_[%s]'%(mm,d))
            # meta_plot_metrics.append('%s-merge_att_align_iou_[%s]'%(mm,d))
            meta_plot_types.append('head')
            # meta_plot_types.append('block')
    # combined meta plots:
    for i in range(len(meta_plot_metrics)):
        print(meta_plot_metrics[i])

        # OLD - all models
        # meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pooled=False, exc_filters='dense')

        # OLD - all models, with pooled average line
        # meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pooled=True, exc_filters='dense')
    
        # OLD - B-16 only plot, taking average score over heads
        # meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pooled=False, exc_filters=['dense', 'RANDOM'], inc_filters='B-16', suff='B-16', base_colors=True, x_block=True, separate_legend=True, no_arch=True)
        
        # B-16 only plot "best_head" - plot results for the "best" head per layer:
        if meta_plot_types[i] == 'head':
            meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pooled=False, exc_filters=['dense', 'RANDOM'], inc_filters='B-16', suff='B-16_best_head', best_head=True, base_colors=True, x_block=True, separate_legend=True, no_arch=True)
            # all models, pertype style plot
            meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pertype=True, exc_filters=['dense', 'RANDOM'], best_head=True, suff='B-16_best_head')


    # # summary table of best results per model
    # for i in range(len(meta_plot_metrics)):
    #     id_ord, all_data = meta_dataload(cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], 'block')
    #     best_block_table(all_data, id_ord, meta_plot_metrics[i])



#################### MAIN ####################



def main(args):
    if args.run_feats:
        extract_feats(args)
    if args.run_met:
        run_metrics(args)
    if args.run_iou:
        run_iou_met(args)
    if args.run_meta:
        run_meta(args)



def parse_args():
    parser = argparse.ArgumentParser('Segmentation Purity metric with COCO maps')
    ######### GENERAL
    parser.add_argument('--overcache', action='store_true', help='disable cache reading but over-write cache when finished')
    parser.add_argument('--nocache', action='store_true', help='fully disable reading and writing of cache files (overrides --overcache)')
    parser.add_argument('--cpu_assembly', action='store_true', help='for use with --dense, gather dense features on cpu instead of gpu to avoid running out of memory')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--mass', type=float, default=0.6, help='when converting attention maps to binary masks, how much attention mass is kept (0.0, 1.0]')
    # RUN MODES
    parser.add_argument('--run_feats', action='store_true', help='extract feature vectors')
    parser.add_argument('--run_met', action='store_true', help='run metrics, must be run after --run_feats')
    parser.add_argument('--run_meta', action='store_true', help='run meta-analysis over pre-cached results')
    parser.add_argument('--run_iou', action='store_true', help='run attention semantic alignment metrics')
    # OUTPUT LOCATIONS
    parser.add_argument('--output_dir', default='seg_analysis_out', help='dir to save metric plots to')
    ######### DATASET
    parser.add_argument('--dataset', default='coco', choices=['coco', 'pin']) # COCO and PartImageNet
    ######### DATASET - COCO
    parser.add_argument('--cocoroot', default='data/coco')
    parser.add_argument('--cocopart', default='val2017', choices=['train2017', 'val2017'])
    parser.add_argument('--imcount', type=int, default=1000, help='how many images to load from the coco dataset')
    ######### DATASET - PARTIMAGENET
    parser.add_argument('--pinroot', default='data/PartImageNet')
    parser.add_argument('--persc', type=int, default=100, help='number of images to load per supercategory in PartImageNet')
    ######### MODEL
    parser.add_argument('--meta_model', default='dino', choices=['dino', 'clip', 'mae', 'timm', 'moco', 'beit', 'random'], help='style of model to load')
    parser.add_argument('--arch', default='B', type=str, choices=['T', 'S', 'B', 'L', 'H'], help='size of vit to load')
    parser.add_argument('--patch', default=16, type=int, help='vit patch size')
    parser.add_argument('--imsize', default=224, type=int, help='image resize size')
    parser.add_argument('--dense', action='store_true', help='enable dense feature extraction mode')
    #########
    args = parser.parse_args()
    return args
    


if __name__ == '__main__':
    args = parse_args()
    main(args)