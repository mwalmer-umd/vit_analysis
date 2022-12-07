"""
###########################################################################
Generic template for analysis scripts

Written by: Matthew Walmer
###########################################################################
"""
import os
import argparse

import numpy as np
import torch
from PIL import Image

from meta_utils.get_model_wrapper import get_model_wrapper
from meta_utils.result_cacher import read_results_cache, save_results_cache
from meta_utils.dense_extractor import dense_extractor
from analysis.attention_plots import meta_plot
from meta_utils.data_summary import best_block_table



#################### PLOTTING ####################



# See existing plotting functions in analysis/attention_plots.py
def dummy_plot(output_dir, mod_id, res, a_m):
    out_name = os.path.join(output_dir, "%s_dummyplot_%s.png"%(mod_id, a_m))  
    return



#################### ANALYSIS METHODS ####################



def dummy_analysis_1(feats):
    # feats will be a tensor of shape:
    # [B,T,F]
    # B = num blocks, T = num tokens, F = feature dim
    return np.zeros([3,3,3])



def dummy_analysis_2(attn):
    # attn will be a tensor of shape:
    # [B,H,T,T]
    # B = num blocks, H = num heads, T = num tokens
    return np.zeros([3,3,3])



#################### ANALYSIS ENGINE ####################



def load_or_run_analysis(args):
    # prep model wrapper
    # if you want to extract the attention maps use:
    # mod_wrap = get_model_wrapper(args.meta_model, args.arch, args.patch, args.imsize, extract_mode='attn', blk_sel='all')
    # if you want to extract the feature maps use:
    mod_wrap = get_model_wrapper(args.meta_model, args.arch, args.patch, args.imsize, extract_mode='feat', blk_sel='all')
    
    # optional - dense wrapper for dense tasks
    # when using the dense wrapper, DO NOT preprocess inputs. Feed PIL.Image objects directly
    # currently only works with extract_mode='feat'
    if args.dense:
        mod_wrap = dense_extractor(mod_wrap, batch_limit=args.batch, cpu_assembly=args.cpu_assembly)

    # prep names for analysis methods
    # each analysis method is given a unique name to aid in cache and plot saving
    analysis_methods = ['dummy_analysis_method_1', 'dummy_analysis_method_2']

    # check cache
    if not (args.nocache or args.overcache):
        results, found, not_found = read_results_cache(mod_wrap.mod_id, analysis_methods)
        if len(results) == len(analysis_methods):
            return results, analysis_methods

    # load model
    mod_wrap.load()

    # prep dataset
    dataset = None # TODO prep dataset wrapper here

    # run metrics
    print('Running Metrics...')
    all_results = {}
    for a_m in analysis_methods:
        all_results[a_m] = []
    for img, lab in dataset:
        # run image
        x = mod_wrap.transform(img) # TODO - disable if using dense mode
        x = torch.unsqueeze(x, 0).to(mod_wrap.device)
        fs = mod_wrap.get_activations(x) # get features
        fs = torch.cat(fs)
        # run metrics
        m1 = dummy_analysis_1(fs)
        m2 = dummy_analysis_2(fs)
        all_results['dummy_analysis_method_1'].append(m1.cpu().numpy())
        all_results['dummy_analysis_method_2'].append(m2.cpu().numpy())

    # stack results
    results = []
    for a_m in analysis_methods:
        res = all_dict[a_m]
        res = np.stack(res, axis=0)
        results.append(res)
    
    # cache results
    if not args.nocache:
        save_results_cache(results, mod_wrap.mod_id, analysis_methods)
    # note size of the results cache depends on the metric:
    # I = num images, B = num blocks, H = num heads
    #   head: [I, B, H]
    #   head_p: [B, H]
    #   block: [I, B]
    #   block_p: [B]
    #     ('_p' = 'pre-pooled' on the image dimension)
    
    return results, analysis_methods



#################### RUNNING MODES ####################



def run_metrics(args):
    results, analysis_methods = load_or_run_analysis(args)
    # get mod_id
    mod_wrap = get_model_wrapper(args.meta_model, args.arch, args.patch, args.imsize, 'feat', 'all')
    mod_id = mod_wrap.mod_id
    if args.dense:
        mod_id += '-dense'
    # make plots
    print('making plots...')
    output_dir = os.path.join(args.output_dir, mod_id)
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(analysis_methods)):
        a_m = analysis_methods[i]
        res = results[i]
        dummy_plot(output_dir, mod_id, res, a_m)



# meta analysis - gather the results for separate models into a single plot
def run_meta(args):
    cache_dir = 'all_results'
    dirs = os.listdir(cache_dir)
    if len(dirs) == 0:
        print('WARNING: can only run meta analysis on cached results. no cached results found')
        return
    dirs.sort()
    print('Found %i cached results'%len(dirs))
    print(dirs)
    # metrics and metric info
    meta_plot_metrics = ['dummy_analysis_method_1', 'dummy_analysis_method_2']
    meta_plot_types = ['head', 'block'] # these specify the format of the cached results
    # meta plots
    meta_out_dir = os.path.join(args.output_dir, '_meta')
    os.makedirs(meta_out_dir, exist_ok=True)
    # combined meta plots
    for i in range(len(meta_plot_metrics)):
        meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pooled=False)
        meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pooled=True)
        meta_plot(meta_out_dir, cache_dir, dirs, meta_plot_metrics[i], meta_plot_types[i], pooled=False, inc_filters='B-16', suff='B-16', base_colors=True)



#################### MAIN ####################



def main(args):
    if args.run_met:
        run_metrics(args)
    if args.run_meta:
        run_meta(args)



def parse_args():
    parser = argparse.ArgumentParser('Generic analysis template')
    ######### GENERAL
    parser.add_argument('--overcache', action='store_true', help='disable cache reading but over-write cache when finished')
    parser.add_argument('--nocache', action='store_true', help='fully disable reading and writing of cache files (overrides --overcache)')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--output_dir', default='seg_analysis_out', help='dir to save metric plots to')
    ######### RUN MODES
    parser.add_argument('--run_met', action='store_true', help='run metrics, must be run after --run_feats')
    parser.add_argument('--run_meta', action='store_true', help='run meta-analysis over pre-cached results')    
    ######### MODEL
    parser.add_argument('--meta_model', default='dino', choices=['dino', 'clip', 'mae', 'timm', 'moco'], help='style of model to load')
    parser.add_argument('--arch', default='B', type=str, choices=['T', 'S', 'B', 'L', 'H'], help='size of vit to load')
    parser.add_argument('--patch', default=16, type=int, help='vit patch size')
    parser.add_argument('--imsize', default=224, type=int, help='image resize size')
    parser.add_argument('--dense', action='store_true', help='enable dense feature extraction mode')
    ######### DATASET
    # add flags as needed
    ######### METRICS
    # add flags as needed
    #########
    args = parser.parse_args()
    return args
    


if __name__ == '__main__':
    args = parse_args()
    main(args)