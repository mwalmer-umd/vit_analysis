"""
###########################################################################
A suite of transformer analysis functions specifically focused on features
extracted from multiple layers of each network.

Written by: Saksham Suri
###########################################################################
"""
import os
import argparse
import time
import colorsys
import shutil
import json
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from imagenet50_dataset import Imagenet50
from meta_utils.get_model_wrapper import get_model_wrapper
from meta_utils.simple_progress import SimpleProgress
from cka import cka_runner
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'sans-serif'})

from PIL import Image, ImageColor
from meta_utils.plotters import pooled_blockwise_comparison_plot, pertype_blockwise_comparison_plot
from meta_utils.plot_format import display_names



#################### MODEL RUNNING ####################



def load_or_run_analysis(args):
    mod_wrap = get_model_wrapper(args.meta_model, args.arch, args.patch, args.imsize, 'none')
    mod_id = mod_wrap.mod_id
    # load model, prep dataloader
    print('loading model...')
    mod_wrap.load()
    train_dataset = Imagenet50(root=args.dataroot, num_samples_per_class=args.perclass, transform=mod_wrap.transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=False)
    
    if(args.run_cka or args.last_layer_dump):
        mod_wrap_2 = get_model_wrapper(args.meta_model_2, args.arch_2, args.patch_2, args.imsize_2, 'none')
        mod_id_2 = mod_wrap_2.mod_id
        # load model, prep dataloader
        print('loading model...')
        mod_wrap_2.load()
        train_dataset_2 = Imagenet50(root=args.dataroot, num_samples_per_class=args.perclass, transform=mod_wrap_2.transform)
        train_dataloader_2 = DataLoader(train_dataset_2, batch_size=args.batch, shuffle=False)
        return (mod_wrap, mod_id, train_dataloader, mod_wrap_2, mod_id_2, train_dataloader_2)
    return (mod_wrap, mod_id, train_dataloader)



def get_last_layer(mod_id):
    if mod_id.split('-')[0] in ['DINO', 'TIMM', 'MAE', 'MOCO', 'BEIT']:
        if('-S-' in mod_id or '-B-' in mod_id):
            mod_layer = ['blocks.11']
        elif('-L-' in mod_id):
            mod_layer = ['blocks.23']
        elif('-H-' in mod_id):
            mod_layer = ['blocks.31']
    else:
        if('-B-' in mod_id):
            mod_layer = ['visual.transformer.resblocks.11']
        elif('-L-' in mod_id):
            mod_layer = ['visual.transformer.resblocks.23']
    return mod_layer



def run_cka_analysis(args):
    mod_wrap, mod_id, train_dataloader, mod_wrap_2, mod_id_2, train_dataloader_2 = load_or_run_analysis(args)
    # run cka analysis
    print('running CKA analysis...')
    cka_runner.run_cka(mod_wrap, mod_id, train_dataloader, mod_wrap_2, mod_id_2, train_dataloader_2, args.output_dir)



def run_dump_norms(args):
    mod_wrap, mod_id, train_dataloader = load_or_run_analysis(args)
    # dump norms of cls, spatial tokens
    print('running Residual analysis...')
    cka_runner.dump_norms(mod_wrap, mod_id, train_dataloader, args.output_dir)



def run_cka_residual_analysis(args):
    mod_wrap, mod_id, train_dataloader = load_or_run_analysis(args)
    # run cka analysis for residuals 
    print('running Residual analysis...')
    cka_runner.run_cka_residual(mod_wrap, mod_id, train_dataloader, args.output_dir)



def run_dump_cls(args):
    mod_wrap, mod_id, train_dataloader = load_or_run_analysis(args)
    # dump cluster metrics for cls tokens across blocks
    print('running cls clustering analysis...')
    cka_runner.cls_dumper(mod_wrap, mod_id, train_dataloader, args.output_dir)



def dump_spat(args):
    mod_wrap, mod_id, train_dataloader = load_or_run_analysis(args)
    # dump cluster metrics for spatial tokens across blocks
    print('running spatial clustering analysis...')
    cka_runner.spat_dumper(mod_wrap, mod_id, train_dataloader, args.output_dir)



def run_last_layer_dump(args):
    mod_wrap, mod_id, train_dataloader, mod_wrap_2, mod_id_2, train_dataloader_2 = load_or_run_analysis(args)
    mod_layer = get_last_layer(mod_id)
    mod_layer_2 = get_last_layer(mod_id_2)
    print('running Residual analysis...')
    cka_runner.run_cka_last_layer(mod_wrap, mod_id, train_dataloader, mod_layer, \
     mod_wrap_2, mod_id_2, train_dataloader_2, mod_layer_2, args.output_dir)



def plot_last_layer(args):
    print('Plotting last layer CKA analysis...')
    subset = ['all', 'cls', 'spatial']
    sub_mapper = {'all': 'All', 'cls': 'CLS', 'spatial': 'Spatial'}
    all_dir = os.path.join(args.output_dir, 'cka_last_layer', 'all')
    cls_dir = os.path.join(args.output_dir, 'cka_last_layer', 'cls')
    spat_dir = os.path.join(args.output_dir, 'cka_last_layer', 'spat')
    plot_dir = os.path.join(args.output_dir, 'cka_last_layer', 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    for d, sub in zip([all_dir, cls_dir, spat_dir], subset):
        pairs = []
        cka_vals = []
        for f in os.listdir(d):
            if(f.endswith('.json') and '336' not in f and '384' not in f):
                with open(os.path.join(d, f), 'r') as fp:
                    data = json.load(fp)
                model1_name = data['model1_name']
                model2_name = data['model2_name']
                cka = data['CKA']
                if([model1_name, model2_name] not in pairs and [model2_name, model1_name] not in pairs):
                    pairs.append([model1_name, model2_name])
                    cka_vals.append(cka)
        
        model_mapper = {}
        if(args.only_b16):
            unique_models = [
            'TIMM-ViT-B-16-224', 'CLIP-ViT-B-16-224', 'DINO-ViT-B-16-224', 
            'MOCO-ViT-B-16-224', 'MAE-ViT-B-16-224', 'BEIT-ViT-B-16-224'
        ]
        else:
            unique_models = [
                'TIMM-ViT-S-32-224', 'TIMM-ViT-S-16-224', 'TIMM-ViT-B-32-224',
                'TIMM-ViT-B-16-224', 'TIMM-ViT-B-8-224', 'TIMM-ViT-L-16-224',
                'CLIP-ViT-B-32-224', 'CLIP-ViT-B-16-224', 'CLIP-ViT-L-14-224',
                'DINO-ViT-S-16-224', 'DINO-ViT-S-8-224', 'DINO-ViT-B-16-224', 'DINO-ViT-B-8-224',
                'MOCO-ViT-S-16-224', 'MOCO-ViT-B-16-224', 
                'MAE-ViT-B-16-224', 'MAE-ViT-L-16-224', 'MAE-ViT-H-14-224',
                'BEIT-ViT-B-16-224', 'BEIT-ViT-L-16-224',
            ]
        for i, model_name in enumerate(unique_models):
            model_mapper[model_name] = i
        heatmap = np.zeros([len(model_mapper), len(model_mapper)])
        
        for i in range(len(pairs)):
            if(pairs[i][0] in model_mapper and pairs[i][1] in model_mapper):
                heatmap[model_mapper[pairs[i][0]], model_mapper[pairs[i][1]]] = cka_vals[i]
                heatmap[model_mapper[pairs[i][1]], model_mapper[pairs[i][0]]] = cka_vals[i]
        
        if(args.only_b16):
            # plot heatmap
            fig, ax = plt.subplots(figsize=(7.5,7))
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=20)
            # change colorbar xticks to be 0-1
            # cbar.set_ticks([0, 1])
            plt.title(f'CKA for {sub_mapper[sub]} tokens', fontsize=30)
            model_display_names = [k for k in sorted(model_mapper, key=model_mapper.get)]
            for i in range(len(model_display_names)):
                    model_display_names[i] = display_names(model_display_names[i], no_arch=True)
            plt.xticks(range(len(model_mapper)), model_display_names, rotation=90, fontsize=25)
            plt.yticks(range(len(model_mapper)), model_display_names, rotation=0, fontsize=25)
        else:
            # plot heatmap
            fig, ax = plt.subplots(figsize=(12,12))
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            # add colorbar
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=20)
            plt.title(f'CKA for {sub_mapper[sub]} tokens', fontsize=25)
            model_display_names = [k for k in sorted(model_mapper, key=model_mapper.get)]
            for i in range(len(model_display_names)):
                    model_display_names[i] = display_names(model_display_names[i])
            plt.xticks(range(len(model_mapper)), model_display_names, rotation=90, fontsize=15)
            plt.yticks(range(len(model_mapper)), model_display_names, rotation=0, fontsize=15)
        ax.invert_yaxis()

        # save plot
        save_dir = d.split('/')[-1]
        if(args.only_b16):
            print(plot_dir, os.path.join(plot_dir, f'heatmap_{save_dir}_b16.png'))
            plt.savefig(os.path.join(plot_dir, f'heatmap_{save_dir}_b16.png'), bbox_inches='tight')
            plt.savefig(os.path.join(plot_dir, f'heatmap_{save_dir}_b16.pdf'), bbox_inches='tight')
            # save svg
            plt.savefig(os.path.join(plot_dir, f'heatmap_{save_dir}_b16.svg'), bbox_inches='tight')
        else:
            print(plot_dir, os.path.join(plot_dir, f'heatmap_{save_dir}.png'))
            plt.savefig(os.path.join(plot_dir, f'heatmap_{save_dir}.png'), bbox_inches='tight')
            plt.savefig(os.path.join(plot_dir, f'heatmap_{save_dir}.pdf'), bbox_inches='tight')
            # save svg
            plt.savefig(os.path.join(plot_dir, f'heatmap_{save_dir}.svg'), bbox_inches='tight')



def plot_clust_metrics(args):
    # plot clsutering metrics using the dumped npz files
    print('Plotting clustering metrics...')
    names = []
    data = []
    base_dir = os.path.join(args.output_dir, 'clust_metrics')
    for f in os.listdir(base_dir):
        if('json' in f and '336' not in f and '384' not in f):
            names.append(f.split('.')[0])
            with open(os.path.join(base_dir, f), 'rb') as fi:
                data.append(json.load(fi))
    data_all = []
    for i, _ in enumerate(data):
        temp_data = []
        for key in data[i].keys():
            temp_data.append(data[i][key][2])
        data_all.append(np.array(temp_data))
    if(args.only_b16):
        pooled_blockwise_comparison_plot(base_dir, data_all, names, 'Purity', pooled=False, base_colors=True, inc_filters='-B-16', suff='B-16', x_block=True, separate_legend=True, no_arch=True)
    else:
        pertype_blockwise_comparison_plot(base_dir, data_all, names, 'Purity')
    data_all = []
    for i, _ in enumerate(data):
        temp_data = []
        for key in data[i].keys():
            temp_data.append(data[i][key][0])
        data_all.append(np.array(temp_data))
    if(args.only_b16):
        pooled_blockwise_comparison_plot(base_dir, data_all, names, 'NMI', pooled=False, base_colors=True, inc_filters='-B-16', suff='B-16', x_block=True, separate_legend=True, no_arch=True)
    else:
        pertype_blockwise_comparison_plot(base_dir, data_all, names, 'NMI')
    data_all = []
    for i, _ in enumerate(data):
        temp_data = []
        for key in data[i].keys():
            temp_data.append(data[i][key][1])
        data_all.append(np.array(temp_data))
    if(args.only_b16):
        pooled_blockwise_comparison_plot(base_dir, data_all, names, 'ARI', pooled=False, base_colors=True, inc_filters='-B-16', suff='B-16', x_block=True, separate_legend=True, no_arch=True)
    else:
        pertype_blockwise_comparison_plot(base_dir, data_all, names, 'ARI')
    print('Plotting clustering metrics...')
    names = []
    data = []
    base_dir = os.path.join(args.output_dir, 'clust_metrics_spatial')
    for f in os.listdir(base_dir):
        if('json' in f and '336' not in f and '384' not in f):
            names.append(f.split('.')[0])
            with open(os.path.join(base_dir, f), 'rb') as fi:
                data.append(json.load(fi))
    data_all = []
    for i, _ in enumerate(data):
        temp_data = []
        for key in data[i].keys():
            temp_data.append(data[i][key][2])
        data_all.append(np.array(temp_data))
    if(args.only_b16):
        pooled_blockwise_comparison_plot(base_dir, data_all, names, 'Purity_Spat', pooled=False, base_colors=True, inc_filters='-B-16', suff='B-16', x_block=True, separate_legend=True, no_arch=True)
    else:
        pertype_blockwise_comparison_plot(base_dir, data_all, names, 'Purity_Spat')

    data_all = []
    for i, _ in enumerate(data):
        temp_data = []
        for key in data[i].keys():
            temp_data.append(data[i][key][0])
        data_all.append(np.array(temp_data))
    if(args.only_b16):
        pooled_blockwise_comparison_plot(base_dir, data_all, names, 'NMI_Spat', pooled=False, base_colors=True, inc_filters='-B-16', suff='B-16', x_block=True, separate_legend=True, no_arch=True)
    else:
        pertype_blockwise_comparison_plot(base_dir, data_all, names, 'NMI_Spat')
    data_all = []
    for i, _ in enumerate(data):
        temp_data = []
        for key in data[i].keys():
            temp_data.append(data[i][key][1])
        data_all.append(np.array(temp_data))
    if(args.only_b16):
        pooled_blockwise_comparison_plot(base_dir, data_all, names, 'ARI_Spat', pooled=False, base_colors=True, inc_filters='-B-16', suff='B-16', x_block=True, separate_legend=True, no_arch=True)
    else:
        pertype_blockwise_comparison_plot(base_dir, data_all, names, 'ARI_Spat')



def plot_norms(args):
    print('Plotting norms...')

    names = []
    data_spat = []
    data_cls = []
    base_dir = os.path.join(args.output_dir, 'residual_analysis', 'norm_files_attn')
    counter = 0
    for f in os.listdir(base_dir):
        if('npz' in f and '336' not in f and '384' not in f):
            names.append(f.split('.')[0])
            file_name = os.path.join(base_dir, f)
            data_spat.append(np.mean(np.array(np.load(file_name, allow_pickle=True)['spat']),1))
            data_cls.append(np.mean(np.array(np.load(file_name, allow_pickle=True)['cls']),1))
            counter += 1
    pooled_blockwise_comparison_plot(base_dir, data_spat, names, 'Attn_Spatial', pooled=False, base_colors=True, inc_filters='-B-16', suff='B-16', x_block=True, separate_legend=True, no_arch=True)
    pooled_blockwise_comparison_plot(base_dir, data_cls, names, 'Attn_Cls', pooled=False, base_colors=True, inc_filters='-B-16', suff='B-16', x_block=True, separate_legend=True, no_arch=True)

    names = []
    data_spat = []
    data_cls = []
    base_dir = os.path.join(args.output_dir, 'residual_analysis', 'norm_files_mlp')
    counter = 0
    for f in os.listdir(base_dir):
        if('npz' in f and '336' not in f and '384' not in f):
            names.append(f.split('.')[0])
            file_name = os.path.join(base_dir, f)
            data_spat.append(np.mean(np.array(np.load(file_name, allow_pickle=True)['spat']),1))
            data_cls.append(np.mean(np.array(np.load(file_name, allow_pickle=True)['cls']),1))
            counter += 1
    pooled_blockwise_comparison_plot(base_dir, data_spat, names, 'MLP_Spatial', pooled=False, base_colors=True, inc_filters='-B-16', suff='B-16', x_block=True, separate_legend=True, no_arch=True)
    pooled_blockwise_comparison_plot(base_dir, data_cls, names, 'MLP_Cls', pooled=False, base_colors=True, inc_filters='-B-16', suff='B-16', x_block=True, separate_legend=True, no_arch=True)



def replot_using_npy(args):
    if(args.replot_residualcka_using_npy):
        root = os.path.join(args.output_dir, 'residual_analysis')
        root = [os.path.join(root, 'cka_files_attn'), os.path.join(root, 'cka_files_mlp')]
    elif(args.replot_cka_using_npy):
        root = [os.path.join(args.output_dir, 'cka_analysis')]
    for r in root:
        for f in tqdm(os.listdir(r)):
            if('npy' in f):
                file_name = os.path.join(r, f)
                hsic_matrix = np.load(file_name, allow_pickle=True)
                fig, ax = plt.subplots()
                plt.imshow(hsic_matrix, origin='lower', cmap='magma')
                if(args.replot_residualcka_using_npy):
                    plt.xlabel('Normal Connection', fontsize=25)
                    plt.ylabel('Skip Connection', fontsize=25)
                if(args.replot_cka_using_npy):
                    model1_name = f.split('x')[0]
                    model2_name = f.split('x')[1].split('.')[0]
                    plt.xlabel(f"{display_names(model2_name, no_arch=args.only_b16)}", fontsize=25)
                    plt.ylabel(f"{display_names(model1_name, no_arch=args.only_b16)}", fontsize=25)
                name = f.split('.')[0]
                if(args.replot_residualcka_using_npy):
                    plt.title(f"{display_names(name, no_arch=args.only_b16)}", fontsize=25)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)

                cbar = plt.colorbar()
                cbar.ax.tick_params(labelsize=12)
                # cbar.set_ticks([0, 0.5, 1])
                plt.tight_layout()
                
                save_path = os.path.join(r, f.split('.')[0] + '.png')
                plt.savefig(save_path, bbox_inches='tight')
                save_path = save_path.replace('.png', '.pdf')
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()



#################### MAIN ####################



def main():
    args = parse_args()
    if args.run_cka:
        run_cka_analysis(args)
    if args.run_cka_residual:
        run_cka_residual_analysis(args)
    if args.dump_norms:
        run_dump_norms(args)
    if args.dump_cls:
        run_dump_cls(args)
    if args.plot_clust:
        plot_clust_metrics(args)
    if args.plot_norms:
        plot_norms(args)
    if args.last_layer_dump:
        run_last_layer_dump(args)
    if args.plot_last_layer:
        plot_last_layer(args)
    if(args.dump_spat):
        dump_spat(args)
    if args.replot_residualcka_using_npy or args.replot_cka_using_npy:
        replot_using_npy(args)



def parse_args():
    parser = argparse.ArgumentParser('Run feature analysis')
    # VIS LOCATIONS
    parser.add_argument('--output_dir', default='feature_analysis_out', help='dir to save metric plots to')
    ######### DATASET
    parser.add_argument('--dataroot', default='data/imagenet/train')
    parser.add_argument('--perclass', type=int, default=100, help='number of samples per class to load with Imagenet50')
    parser.add_argument('--batch', type=int, default=8)
    ######### MODEL
    parser.add_argument('--meta_model', default='dino', choices=['dino', 'clip', 'mae', 'timm', 'moco', 'beit', 'random'], help='style of model to load')
    parser.add_argument('--arch', default='B', type=str, choices=['T', 'S', 'B', 'L', 'H'], help='size of vit to load')
    parser.add_argument('--patch', default=16, type=int, help='vit patch size')
    parser.add_argument('--imsize', default=224, type=int, help='image resize size')
    
    parser.add_argument('--meta_model_2', default='dino', choices=['dino', 'clip', 'mae', 'timm', 'moco', 'beit'], help='style of model to load')
    parser.add_argument('--arch_2', default='B', type=str, choices=['T', 'S', 'B', 'L', 'H'], help='size of vit to load')
    parser.add_argument('--patch_2', default=16, type=int, help='vit patch size')
    parser.add_argument('--imsize_2', default=224, type=int, help='image resize size')

    ######### RUN
    parser.add_argument('--run_cka_residual', action='store_true', help='run residual connect CKA analysis')
    parser.add_argument('--dump_norms', action='store_true', help='dump norm ratios for residual analysis')
    parser.add_argument('--run_cka', action='store_true', help='run CKA analysis')
    parser.add_argument('--dump_cls', action='store_true', help='dump cls token features across blocks')
    parser.add_argument('--dump_spat', action='store_true', help='dump cls token features across blocks')

    parser.add_argument('--last_layer_dump', action='store_true', help='CKA only on last layer of each model')
    parser.add_argument('--plot_clust', action='store_true', help='Plot clustering metrics')
    parser.add_argument('--plot_norms', action='store_true', help='Plot ratio of norms between skipp connection and normal path')
    parser.add_argument('--plot_last_layer', action='store_true', help='Plot last layer CKA')
    parser.add_argument('--replot_residualcka_using_npy', action='store_true', help='Plot last layer CKA')
    parser.add_argument('--replot_cka_using_npy', action='store_true', help='Plot last layer CKA')
    parser.add_argument('--only_b16', action='store_true', help='Plot for only B 16 architecture')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    main()