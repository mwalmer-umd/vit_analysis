"""
###########################################################################
Centered Kernal Analysis (CKA) for ViT features.

Written by: Saksham Suri
###########################################################################
"""
from .cka_main import CKA, GetTok
import json
import torch
import os
import numpy as np
from .utils import get_layers_residual, get_layers_cka, get_keep_list, cluster_metric


def run_cka(mod_wrap, mod_id, train_dataloader, mod_wrap_2, mod_id_2, train_dataloader_2, output_dir):
    mod_layer_names_ = get_layers_cka(mod_id)
    mod_layer_names_2_ = get_layers_cka(mod_id_2)
    mod_layer_names = []
    # need to filter out attn_holder for clip layers
    for l in mod_layer_names_:
        if 'holder' not in l:
            mod_layer_names.append(l)
    mod_layer_names_2 = []
    for l in mod_layer_names_2_:
        if 'holder' not in l:
            mod_layer_names_2.append(l)
    name = mod_id + 'x' + mod_id_2 + '.png'
    print(name)

    cka = CKA(mod_wrap.model, mod_wrap_2.model,
              runner1=mod_wrap.extractor.runner,
              runner2=mod_wrap_2.extractor.runner,
              model1_name=mod_id,
              model2_name=mod_id_2,   
              model1_layers=mod_layer_names,
              model2_layers=mod_layer_names_2,
              device='cuda')
    with torch.no_grad():
        cka.compare(train_dataloader, train_dataloader_2)
    
    save_path = os.path.join(output_dir, 'cka_analysis', name)
    os.makedirs(os.path.join(output_dir, 'cka_analysis'), exist_ok=True)
    cka.plot_results(save_path=save_path)

def run_cka_residual(mod_wrap, mod_id, train_dataloader, output_dir):
    mod_layers_identity_attn = get_layers_residual(mod_id, ltype = 'attn_residual')
    mod_layers_normal_attn = get_layers_residual(mod_id, ltype = 'attn_normal')
    mod_layers_identity_mlp = get_layers_residual(mod_id, ltype = 'mlp_residual')
    mod_layers_normal_mlp = get_layers_residual(mod_id, ltype = 'mlp_normal')

    name = mod_id
    print(name)
    # cka analysis on attn residuals
    cka = CKA(mod_wrap.model, mod_wrap.model,
              runner1=mod_wrap.extractor.runner,
              runner2=mod_wrap.extractor.runner,
              model1_name=mod_id,
              model2_name=mod_id,   
              model1_layers=mod_layers_identity_attn,
              model2_layers=mod_layers_normal_attn,
              device='cuda')
    with torch.no_grad():
        cka.compare(train_dataloader, train_dataloader)
    save_path = os.path.join(output_dir, 'residual_analysis', 'cka_files_attn', name)
    os.makedirs(os.path.join(output_dir, 'residual_analysis','cka_files_attn'), exist_ok=True)
    cka.plot_results_residual(save_path=save_path)

    # cka analysis on mlp residuals
    cka = CKA(mod_wrap.model, mod_wrap.model,
              runner1=mod_wrap.extractor.runner,
              runner2=mod_wrap.extractor.runner,
              model1_name=mod_id,
              model2_name=mod_id,   
              model1_layers=mod_layers_identity_mlp,
              model2_layers=mod_layers_normal_mlp,
              device='cuda')
    with torch.no_grad():
        cka.compare(train_dataloader, train_dataloader)
    save_path = os.path.join(output_dir, 'residual_analysis', 'cka_files_mlp', name)
    os.makedirs(os.path.join(output_dir, 'residual_analysis','cka_files_mlp'), exist_ok=True)
    cka.plot_results_residual(save_path=save_path)


def dump_norms(mod_wrap, mod_id, train_dataloader, output_dir):
    mod_layers_identity_attn = get_layers_residual(mod_id, ltype = 'attn_residual')
    mod_layers_normal_attn = get_layers_residual(mod_id, ltype = 'attn_normal')
    mod_layers_identity_mlp = get_layers_residual(mod_id, ltype = 'mlp_residual')
    mod_layers_normal_mlp = get_layers_residual(mod_id, ltype = 'mlp_normal')

    name = mod_id
    print(name)

    # dumping residual norms for attn residuals
    cka = CKA(mod_wrap.model, mod_wrap.model,
              runner1=mod_wrap.extractor.runner,
              runner2=mod_wrap.extractor.runner,
              model1_name=mod_id,
              model2_name=mod_id,   
              model1_layers=mod_layers_identity_attn,
              model2_layers=mod_layers_normal_attn,
              device='cuda')
    with torch.no_grad():
        cka.compare_residual(train_dataloader, train_dataloader)
    save_path = os.path.join(output_dir, 'residual_analysis', 'norm_files_attn', name)
    os.makedirs(os.path.join(output_dir, 'residual_analysis','norm_files_attn'), exist_ok=True)
    np.savez(save_path, all_tokens=cka.all_norm, cls=cka.cls_norm, spat=cka.spat_norm)

    # dumping residual norms for mlp residuals
    cka = CKA(mod_wrap.model, mod_wrap.model,
              runner1=mod_wrap.extractor.runner,
              runner2=mod_wrap.extractor.runner,
              model1_name=mod_id,
              model2_name=mod_id,   
              model1_layers=mod_layers_identity_mlp,
              model2_layers=mod_layers_normal_mlp,
              device='cuda')
    with torch.no_grad():
        cka.compare_residual(train_dataloader, train_dataloader)
    save_path = os.path.join(output_dir, 'residual_analysis', 'norm_files_mlp', name)
    os.makedirs(os.path.join(output_dir, 'residual_analysis','norm_files_mlp'), exist_ok=True)
    np.savez(save_path, all_tokens=cka.all_norm, cls=cka.cls_norm, spat=cka.spat_norm)

def cls_dumper(mod_wrap, mod_id, train_dataloader, output_dir):
    keep_list = get_keep_list(mod_id)
    name = mod_id

    if('RANDOM' not in name):
        token_op = GetTok(mod_wrap.model,
                runner1=mod_wrap.extractor.runner,
                model1_name=mod_id,
                model1_layers=keep_list,
                mode = 'cls',
                device='cuda')

        with torch.no_grad():
            token_op.dump_token(train_dataloader)
        metrics = {}
        for layer in keep_list:
            nmi,ari, pur = cluster_metric(50, token_op.feats[layer], token_op.labels)
            metrics[layer] = [nmi, ari, pur]
    else:
        # create labels 100 for each class from 0-49
        labels = np.repeat(np.arange(50), 100)
        metrics = {}
        for layer in keep_list:
            nmi,ari, pur = cluster_metric(50, np.random.rand(5000, 768), labels)
            metrics[layer] = [nmi, ari, pur]
        print(metrics)
        
    save_path = os.path.join(output_dir, 'clust_metrics', name + '.json')
    os.makedirs(os.path.join(output_dir, 'clust_metrics'), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(metrics, f)

def spat_dumper(mod_wrap, mod_id, train_dataloader, output_dir):
    keep_list = get_keep_list(mod_id)
    name = mod_id

    if('RANDOM' not in name):
        token_op = GetTok(mod_wrap.model,
                runner1=mod_wrap.extractor.runner,
                model1_name=mod_id,
                model1_layers=keep_list,
                mode = 'spat',
                device='cuda')

        with torch.no_grad():
            token_op.dump_token(train_dataloader)
        metrics = {}
        for layer in keep_list:
            nmi,ari, pur = cluster_metric(50, token_op.feats[layer], token_op.labels)
            metrics[layer] = [nmi, ari, pur]
    else:
        # create labels 100 for each class from 0-49
        labels = np.repeat(np.arange(50), 100)
        metrics = {}
        for layer in keep_list:
            nmi,ari, pur = cluster_metric(50, np.random.rand(5000, 768), labels)
            metrics[layer] = [nmi, ari, pur]
        print(metrics)
        
    save_path = os.path.join(output_dir, 'clust_metrics_spatial', name + '.json')
    os.makedirs(os.path.join(output_dir, 'clust_metrics_spatial'), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(metrics, f)

def run_cka_last_layer(mod_wrap, mod_id, train_dataloader, mod_layer_names, mod_wrap_2, mod_id_2, train_dataloader_2, mod_layer_names_2, output_dir):
    name = mod_id + 'x' + mod_id_2 + '.json'

    os.makedirs(os.path.join(output_dir, 'cka_last_layer'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'cka_last_layer', 'all'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'cka_last_layer', 'cls'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'cka_last_layer', 'spat'), exist_ok=True)

    for mode in ['all', 'cls', 'spat']:
        save_path = os.path.join(output_dir, 'cka_last_layer', mode, name)
        if(not os.path.exists(save_path)):
            cka = CKA(mod_wrap.model, mod_wrap_2.model,
                    runner1=mod_wrap.extractor.runner,
                    runner2=mod_wrap_2.extractor.runner,
                    model1_name=mod_id,
                    model2_name=mod_id_2,   
                    model1_layers=mod_layer_names,
                    model2_layers=mod_layer_names_2,
                    device='cuda')
            with torch.no_grad():
                cka.compare(train_dataloader, train_dataloader_2, mode)

            with open(save_path, 'w') as f:
                json.dump(cka.export(), f)