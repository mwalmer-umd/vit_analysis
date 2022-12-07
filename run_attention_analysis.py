"""
###########################################################################
A suite of transformer analysis functions specifically focused on attention
maps produced by multi-headed self-attention layers. This code is partially
based on code from the original DINO repo, specifically portions that
export visualizations of the raw attention maps.

Written by: Matthew Walmer
###########################################################################
"""
import os
import argparse
import time
import shutil

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from imagenet50_dataset import Imagenet50
from supergrid import make_supergrid
from meta_utils.get_model_wrapper import get_model_wrapper
from meta_utils.simple_progress import SimpleProgress
from analysis.attention_metrics import *
from analysis.attention_plots import *



#################### METRICS ####################



def load_or_run_analysis(args, analysis_methods):
    mod_wrap = get_model_wrapper(args.meta_model, args.arch, args.patch, args.imsize, 'attn')
    mod_id = mod_wrap.mod_id
    # check for cached existing results
    if not args.nocache and not args.overcache:
        print('checking for cache files...')
        results = {}
        have_cache = True
        for a_m in analysis_methods:
            fname = os.path.join(args.output_dir, 'cache', mod_id, "%s_%s_cache.npy"%(mod_id, a_m))
            if os.path.isfile(fname):
                results[a_m] = np.load(fname)
            else:
                have_cache = False
                break
        if have_cache:
            print('cache files found')
            return results
        else:
            print('cache files not found')
            results = {}
    # load model, prep dataloader
    print('loading model...')
    mod_wrap.load()
    train_dataset = Imagenet50(root=args.dataroot, num_samples_per_class=args.perclass, transform=mod_wrap.transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=False)
    # prep metric runners
    metric_runners = []
    for a_m in analysis_methods:
        # attention distance
        if a_m == 'avg-att-dist':
            MR = AttentionDistance()
        # amount of spatial-token-attention devoted to cls token
        elif a_m == 'spatial-cls-att':
            MR = SpatialCLSAttention()
        # aggregated attention maps
        elif a_m == 'avg-att-on-token':
            MR = AvgAttentionOnToken()
        elif a_m == 'avg-spc-att-on-token':
            MR = AvgAttentionOnToken('spc')
        elif a_m == 'avg-spcpure-att-on-token':
            MR = AvgAttentionOnToken('spcpure')
        elif a_m == 'avg-cls-att-on-token':
            MR = AvgAttentionOnToken('cls')
        # per-class aggregated attention maps
        elif a_m == 'avg-att-on-token-per-class':
            MR = AvgAttentionOnTokenPerClass(50)
        # alligned aggregated attention maps
        elif a_m == 'avg-aligned-att-on-token':
            MR = AvgAlignedAttentionOnToken()
        # stdev metrics
        elif a_m == 'stdev-over-token-pos':
            MR = PositionDeviation()
        elif a_m == 'stdev-over-head':
            MR = HeadDeviation()
        # experiment: attention patterns as semantic shortcuts
        elif a_m == 'att-pat-shortcuts':
            MR = AttentionPatternShortcuts()
        else:
            print('ERROR: unknown analysis method: ' + a_m)
            exit(-1)
        metric_runners.append(MR)
    # get activations for all images
    print('Running images...')
    SP = SimpleProgress(end=args.perclass*50, step=args.batch)
    for img, label in train_dataloader:
        SP.update()
        # run images, get attentions
        attentions = mod_wrap.get_activations(img)
        attentions = torch.stack(attentions, dim=1).detach()
        # run analysis methods:
        for MR in metric_runners:
            MR.add(attentions, label)
        attentions = None # free memory
    SP.finish()  
    # pack results
    results = {}
    for MR in metric_runners:
        results[MR.metric_name] = MR.get_results()
    # cache results
    if not args.nocache:
        os.makedirs(os.path.join(args.output_dir, 'cache', mod_id), exist_ok=True)
        for a_m in analysis_methods:
            fname = os.path.join(args.output_dir, 'cache', mod_id, "%s_%s_cache.npy"%(mod_id, a_m))
            np.save(fname, results[a_m])
    return results



# extra caching helper for caching post-processed metrics at the end
def extra_cache(args, mod_id, results, analysis_methods):
    if args.nocache: return
    for a_m in analysis_methods:
        fname = os.path.join(args.output_dir, 'cache', mod_id, "%s_%s_cache.npy"%(mod_id, a_m))
        np.save(fname, results[a_m])



def run_metrics(args):
    # prepare dir
    mod_wrap = get_model_wrapper(args.meta_model, args.arch, args.patch, args.imsize, 'attn')
    mod_id = mod_wrap.mod_id
    print(mod_id)
    mod_wrap = None
    full_output_dir = os.path.join(args.output_dir, mod_id)
    os.makedirs(full_output_dir, exist_ok=True)

    # analysis methods
    block_methods = ['stdev-over-head']
    head_methods = ['avg-att-dist', 'spatial-cls-att', 'stdev-over-token-pos']
    token_methods = ['avg-att-on-token', 'avg-spc-att-on-token', 'avg-spcpure-att-on-token', 'avg-cls-att-on-token', 'avg-aligned-att-on-token']
    perclass_token_methods = []

    # RETIRED METRICS
    # block_methods = []
    # head_methods = ['att-pat-shortcuts']
    # token_methods = []
    # perclass_token_methods = ['avg-att-on-token-per-class']

    # run or load results
    analysis_methods = block_methods + head_methods + token_methods + perclass_token_methods
    results = load_or_run_analysis(args, analysis_methods)
    
    # post-processing metrics
    post_proc_methods = []
    if 'avg-att-on-token-per-class' in analysis_methods:
        res = deviation_by_class(results['avg-att-on-token-per-class'])
        results['stdev-over-class'] = res
        head_methods.append('stdev-over-class')
        post_proc_methods.append('stdev-over-class')
    if 'avg-aligned-att-on-token' in analysis_methods:
        d, cx, cy = average_att_offset(results['avg-aligned-att-on-token'])
        results['avg-att-offset'] = d
        head_methods.append('avg-att-offset')
        post_proc_methods.append('avg-att-offset')
        # specialized vs-plot for attention offsets:
        temp_res = {}
        temp_res['avg-x-offset'] = cx
        temp_res['avg-y-offset'] = cy
        vs_plot(full_output_dir, mod_id, temp_res, 'avg-x-offset', 'avg-y-offset', fs=20, fn_override='average-att-offset-vis')
    if len(post_proc_methods) > 0:
        extra_cache(args, mod_id, results, post_proc_methods) # for meta analysis

    print('Generating Plots...')
    t0 = time.time()

    # block-level metrics
    print('Block-Level Plots')
    for a_m in block_methods:
        print('  ' + a_m)
        block_level_plots(full_output_dir, mod_id, results[a_m], a_m, sbars=False)

    # per-head metric plots
    print('Head-Level Plots')
    for a_m in head_methods:
        print('  ' + a_m)
        # head_level_plots(full_output_dir, mod_id, results[a_m], a_m, sub_sel=[0,1,2,3,7,8,9,10]) # optional - subsampling
        head_level_plots(full_output_dir, mod_id, results[a_m], a_m, sbars=False)

    # model summary plots - visualize the whole model colorized based on per-head metrics
    print('Head-Level Summary Plots')
    srt_ord = None
    for a_m in head_methods:
        print('  ' + a_m)
        # option 1: retain the same ordering as the first plot
        # srt_ord = summary_plot(full_output_dir, mod_id, results[a_m], a_m, sort_order=True, pre_order=srt_ord)
        # option 2: use sorted order (likely different order) for all plots
        _ = summary_plot(full_output_dir, mod_id, results[a_m], a_m, sort_order=True)
        # option 3: do not sort order, keep original head order in plots
        _ = summary_plot(full_output_dir, mod_id, results[a_m], a_m, sort_order=False)

    # token plots - metrics computed at the per-token level
    print('Token-Level Plots')
    for a_m in token_methods:
        print('  ' + a_m)
        pre_shaped=False
        if a_m == 'avg-aligned-att-on-token':
            pre_shaped=True
        token_plot(full_output_dir, mod_id, results[a_m], a_m, pre_scale=False, pre_shaped=pre_shaped)
        token_plot(full_output_dir, mod_id, results[a_m], a_m+'-[PRE-SCALED]', pre_scale=True, pre_shaped=pre_shaped)
        # NOTE - optional plots that plot the CLS token next to the SPC tokens - causes value scaling issues
        if a_m != 'avg-aligned-att-on-token':
            token_plot(full_output_dir, mod_id, results[a_m], a_m+'-[WITH-CLS]', pre_scale=False, pre_shaped=pre_shaped, include_cls=True)
            token_plot(full_output_dir, mod_id, results[a_m], a_m+'-[PRE-SCALED][WITH-CLS]', pre_scale=True, pre_shaped=pre_shaped, include_cls=True)
    
    # per-class token plots - metrics computed at the per-token-per-class level
    print('Per-Class Token-Level Plots')
    for a_m in perclass_token_methods:
        print('  ' + a_m)
        full_output_dir_perclass = os.path.join(full_output_dir, a_m)
        os.makedirs(full_output_dir_perclass, exist_ok=True)
        for i in range(50):
            a_m_i = '%s-%02i'%(a_m, i)
            res_i = results[a_m][i]
            token_plot(full_output_dir_perclass, mod_id, res_i, a_m_i, pre_scale=False)
        full_output_dir_perclass = os.path.join(full_output_dir, a_m+'-[PRE-SCALED]')
        os.makedirs(full_output_dir_perclass, exist_ok=True)
        for i in range(50):
            a_m_i = '%s-%02i'%(a_m, i)
            res_i = results[a_m][i]
            token_plot(full_output_dir_perclass, mod_id, res_i, a_m_i+'-[PRE-SCALED]', pre_scale=True)

    # metric vs metric plots (for per-head metrics)
    print('VS-Plots')
    for i in range(len(head_methods)):
        for j in range(len(head_methods)-i-1):
            am1 = head_methods[i]
            am2 = head_methods[i+j+1]
            vs_plot(full_output_dir, mod_id, results, am1, am2)
    
    print('Done in %.2s seconds'%(time.time()-t0))



def run_meta_analysis(args):
    cache_dir = os.path.join(args.output_dir, 'cache')
    dirs = os.listdir(cache_dir)
    if len(dirs) == 0:
        print('WARNING: can only run meta analysis on cached results. no cached results found')
        return
    dirs.sort()
    print('Found %i cached results'%len(dirs))
    print(dirs)

    ##### COMPUTE RANGES #####
    # global min/max per metric
    analysis_methods = ['avg-att-dist', 'spatial-cls-att', 'avg-att-offset']
    for a_m in analysis_methods:
        a_m_min = None
        a_m_max = None
        for mod_id in dirs:
            fname = os.path.join(args.output_dir, 'cache', mod_id, "%s_%s_cache.npy"%(mod_id, a_m))
            results = np.load(fname)
            results = np.mean(results, axis=0)
            r_min = np.min(results)
            r_max = np.max(results)
            if a_m_min is None:
                a_m_min = r_min
                a_m_max = r_max
            a_m_min = min([r_min, a_m_min])
            a_m_max = max([r_max, a_m_max])
        print(a_m)
        print('  min: %.5f'%a_m_min)
        print('  max: %.5f'%a_m_max)
    
    ##### GATHER PLOTS #####
    # gather figures by type
    print('grouping figures by type')
    temp_dir = os.path.join(args.output_dir, dirs[0])
    figures = os.listdir(temp_dir)
    figs = []
    for f in figures:
        if os.path.isdir(os.path.join(temp_dir, f)): # ignore sub-dirs
            continue
        f = f.replace(dirs[0],'').replace('.png','')[1:]
        figs.append(f)
    figs.sort()
    print('Found %i figure types'%len(figs))
    print(figs)
    for f in figs:
        fig_out_dir = os.path.join(args.output_dir, '_grouped', f)
        os.makedirs(fig_out_dir, exist_ok=True)
        for mod_id in dirs:
            fn = '%s_%s.png'%(mod_id, f)
            src = os.path.join(args.output_dir, mod_id, fn)
            dst = os.path.join(fig_out_dir, fn)
            # speed up: check for existing files
            if os.path.isfile(dst):
                continue
            try:
                shutil.copy(src, dst)
            except:
                print('ERROR: could not find file: %s'%src)

    # ##### META PLOTS #####
    # combine plots of head-level and block-level metrics
    meta_plot_metrics = ['stdev-over-head', 'avg-att-dist', 'spatial-cls-att', 'stdev-over-token-pos', 'avg-att-offset']
    meta_plot_types = ['block', 'head', 'head', 'head', 'head_p']
    # meta_plot_metrics = ['avg-att-dist']
    # meta_plot_types = ['head']
    meta_out_dir = os.path.join(args.output_dir, '_meta')
    os.makedirs(meta_out_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, 'cache')
    # single metric plots
    print('generating meta plots...')
    for i in range(len(meta_plot_metrics)):
        print(meta_plot_metrics[i])
        # OLD: all models, no pooling
        # meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pooled=False)
        # OLD: all models, with pooling by type
        # meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pooled=True)
        # B-16 models only
        meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pooled=False, inc_filters='B-16', suff='B-16', base_colors=True, x_block=True, separate_legend=True, no_arch=True)
        # B-16 models only, breakout style plot
        meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pooled=False, inc_filters='B-16', suff='B-16', base_colors=True, x_block=True, separate_legend=True, no_arch=True, breakout=True)
        # all models, pertype style plot
        meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pertype=True)

    # meta-vs-plots: for pooled network-level metrics
    for i in range(len(meta_plot_metrics)):
        for j in range(len(meta_plot_metrics)-i-1):
            a_m1 = meta_plot_metrics[i]
            a_m1_type = meta_plot_types[i]
            a_m2 = meta_plot_metrics[i+j+1]
            a_m2_type = meta_plot_types[i+j+1]
            meta_vs_plot(meta_out_dir, cache_dir, dirs, a_m1, a_m1_type, a_m2, a_m2_type)



#################### ATTENTION MAP VISUALIZATIONS ####################



# NOTE - 'spcagg' is a special mode that instead averages together all 
# spatial token attention maps
# NOTE - in "diagonal" mode, will sample all spatial tokens along the
# top left to bottom right diagonal
def select_positions(tok_h, tok_w, diag=False):
    if not diag:
        pos_names = ['cls', 'top', 'left', 'center', 'right', 'bottom', 'spcagg']
        if tok_h % 2 == 0:
            mid_h = int(tok_h/2)
        else:
            mid_h = int((tok_h+1)/2)
        if tok_w % 2 == 0:
            mid_w = int(tok_w/2)
        else:
            mid_w = int((tok_w+1)/2)
        p_top = mid_w
        p_left = (mid_h-1)*tok_w + 1
        p_center = (mid_h-1)*tok_w + mid_w
        p_right = mid_h*tok_w
        p_bottom = (tok_h-1)*tok_w + mid_w
        positions = [0, p_top, p_left, p_center, p_right, p_bottom, -1]
    else:
        pos_names = []
        positions = []
        for i in range(tok_h):
            p = (tok_w + 1)*i + 1
            positions.append(p)
            pos_names.append('diag%02i'%i)
    return positions, pos_names



def export_attention_maps(args):
    t0 = time.time()
    mod_wrap = get_model_wrapper(args.meta_model, args.arch, args.patch, args.imsize, 'attn')
    mod_wrap.load()
    mod_id = mod_wrap.mod_id
    dump_dir = os.path.join(args.vis_dump, mod_id)
    tok_h = int(args.imsize/args.patch)
    tok_w = int(args.imsize/args.patch)
    positions, pos_names = select_positions(tok_h, tok_w)
    print('exporting attention maps to: ' + dump_dir)
    os.makedirs(dump_dir, exist_ok=True)
    print('loading image from: ' + args.vis_in)
    image_list = os.listdir(args.vis_in)
    print('found %i images'%len(image_list))
    for imgf in image_list:
        print(imgf)
        # load and prep image
        fname = os.path.join(args.vis_in, imgf)
        img = Image.open(fname)
        img = img.convert('RGB')
        img_np = np.array(img)
        orig_size = [img_np.shape[1], img_np.shape[0]]
        img = mod_wrap.transform(img).unsqueeze(0)
        # copy image to dump dir
        dst_fname = os.path.join(dump_dir, imgf)
        shutil.copy(fname, dst_fname)
        # run image
        attns = mod_wrap.get_activations(img)
        attns = torch.stack(attns, dim=1).detach()
        attns = attns.cpu().numpy()
        # fix for CLIP - PIL will not take float16
        if attns.dtype == np.float16:
            attns = attns.astype(np.float32)
        nblks = attns.shape[1]
        nheads = attns.shape[2]
        # export images
        for b in range(nblks):
            print('  block %i'%b)
            for h in range(nheads):
                for p in range(len(positions)):
                    pos_name = pos_names[p]
                    pidx = positions[p]
                    attn_img = get_attn_img(attns, orig_size, b, h, pidx, tok_h, tok_w)
                    out_name = "%s+attention+%s+attn%02i+blk%02i.png"%(imgf, pos_name, h, b)
                    out_name = os.path.join(dump_dir, out_name)
                    plt.imsave(fname=out_name, arr=attn_img, format='png')
    print('done in %.1f seconds'%(time.time()-t0))



# NOTE - if pidx == -1, this will return the average of all spatial
# attention maps instead. This is for 'spcagg' mode (spatial aggregated)
def get_attn_img(attns, orig_size, blk, head, pidx, tok_h, tok_w):
    if pidx == -1:
        attn = np.mean(attns[0, blk, head, 1:, 1:], axis=0)
    else:
        attn = attns[0, blk, head, pidx, 1:]
    attn = np.reshape(attn, [tok_h, tok_w])
    attn = Image.fromarray(attn)
    attn = attn.resize(orig_size, resample=Image.NEAREST)
    attn = np.array(attn)
    return attn



def make_attention_grids(args):
    mod_wrap = get_model_wrapper(args.meta_model, args.arch, args.patch, args.imsize, 'attn')
    mod_id = mod_wrap.mod_id
    dump_dir = os.path.join(args.vis_dump, mod_id)
    if not os.path.isdir(dump_dir):
        print('WARNING: Could not find attention dumps at ' + dump_dir)
        exit(-1)
    output_dir = os.path.join(args.vis_out, mod_id)
    print('Saving grid images to: ' + output_dir)
    # default supergrid configs:
    out_pb = os.path.join(output_dir, 'pos-v-blk')
    out_hp = os.path.join(output_dir, 'head-v-pos')
    image_list = os.listdir(args.vis_in)
    for img in image_list:
        # pos vs block
        make_supergrid(dump_dir, out_pb, xaxis='pos', yaxis='blk', use_all='head', img=img, mod_id=mod_id)
        # head vs pos
        make_supergrid(dump_dir, out_hp, xaxis='head', yaxis='pos', use_all='blk', img=img, mod_id=mod_id)
        # head vs block
        make_supergrid(dump_dir, out_hp, xaxis='head', yaxis='blk', use_all='pos', img=img, mod_id=mod_id)



#################### MAIN ####################



def main():
    args = parse_args()
    if args.run_met:
        run_metrics(args)
    if args.run_exp:
        export_attention_maps(args)
    if args.run_grids:
        make_attention_grids(args)
    if args.run_meta:
        run_meta_analysis(args)



def parse_args():
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    ######### GENERAL
    parser.add_argument('--nocache', action='store_true', help='disable reading and writing of cache files')
    parser.add_argument('--overcache', action='store_true', help='disable cache reading but over-write cache when finished')
    # RUN MODES
    parser.add_argument('--run_met', action='store_true', help='run metrics')
    parser.add_argument('--run_exp', action='store_true', help='export attention map visualizations')
    parser.add_argument('--run_grids', action='store_true', help='generate grids of attention images, must do --run_exp first')
    parser.add_argument('--run_meta', action='store_true', help='run meta-analysis over pre-cached results')
    # VIS LOCATIONS
    parser.add_argument('--output_dir', default='attention_analysis_out', help='dir to save metric plots to')
    parser.add_argument('--vis_in', default='vis_in', help='dir of input images to run visualizations on')
    parser.add_argument('--vis_dump', default='vis_dump', help='dir to dump intermediate vis files')
    parser.add_argument('--vis_out', default='vis_out', help='dir to output visualization grid files to')
    ######### DATASET
    parser.add_argument('--dataroot', default='data/imagenet/train')
    parser.add_argument('--perclass', type=int, default=100, help='number of samples per class to load with Imagenet50')
    parser.add_argument('--batch', type=int, default=2)
    ######### MODEL
    parser.add_argument('--meta_model', default='dino', choices=['dino', 'clip', 'mae', 'timm', 'moco', 'beit'], help='style of model to load')
    parser.add_argument('--arch', default='B', type=str, choices=['T', 'S', 'B', 'L', 'H'], help='size of vit to load')
    parser.add_argument('--patch', default=16, type=int, help='vit patch size')
    parser.add_argument('--imsize', default=224, type=int, help='image resize size')
    args = parser.parse_args()
    if not args.run_met and not args.run_exp and not args.run_grids and not args.run_meta:
        print('WARNING: Must specify at least one of: --run_met --run_exp --run_grids --run_meta')
    return args



if __name__ == '__main__':
    main()