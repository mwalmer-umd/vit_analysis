"""
###########################################################################
Gather and plot downstream analysis results

Written by: Saksham Suri
###########################################################################
""" 
import pickle
import os
import csv
import matplotlib.pyplot as plt
from meta_utils.plotters import pooled_blockwise_comparison_plot, pertype_blockwise_comparison_plot
from meta_utils.data_summary import best_block_table
import re
import numpy as np

root_dir = 'all_results/'
os.makedirs('all_results/plots', exist_ok=True)
def plot_knn(topk = 'top1'):
    print('Plotting knn metrics...')
    names = []
    data = []
    root_dir = 'all_results'
    for mod_id in sorted(os.listdir(root_dir)):
        if mod_id!='plots' and mod_id!='davis_dense' and mod_id!='plots_all':
            names.append(mod_id)
            mod_dir = os.path.join(root_dir, mod_id)
            knn_op = os.listdir(mod_dir)
            knn_op = [x for x in knn_op if 'knn' in x]
            knn_op.sort(key=lambda f: int(re.sub('\D', '', f)))
            temp_knn = []
            for op in knn_op:
                op_dir = os.path.join(mod_dir, op)
                with open(op_dir, 'rb') as f:
                    knn_metrics = pickle.load(f)
                temp_knn.append(knn_metrics[20][topk])
            data.append(np.array(temp_knn))
    pooled_blockwise_comparison_plot('all_results/plots', data, names, f'KNN_{topk}_Acc', \
    pooled=False, base_colors=True, inc_filters='-B-16', suff='B-16', x_block=True, separate_legend=True, no_arch=True)
    best_block_table(data, names, topk, higher_better=True)

plot_knn('top1')
plot_knn('top5')

def plot_spair(metric = 'pck@0.1'):
    print('Plotting keypoint metrics...')
    names = []
    data = []
    root_dir = 'all_results'
    for mod_id in sorted(os.listdir(root_dir)):
        if mod_id!='plots' and mod_id!='davis_dense' and mod_id!='plots_all':
            names.append(mod_id)
            mod_dir = os.path.join(root_dir, mod_id)
            kp_op = os.listdir(mod_dir)
            kp_op = [x for x in kp_op if 'spair' in x]
            kp_op.sort(key=lambda f: int(re.sub('\D', '', f)))
            temp_knn = []
            for op in kp_op:
                op_dir = os.path.join(mod_dir, op)
                with open(op_dir, 'rb') as f:
                    kp_metrics = pickle.load(f)
                temp_knn.append(100*np.mean(kp_metrics[metric]))
            data.append(np.array(temp_knn))
    pooled_blockwise_comparison_plot('all_results/plots', data, names, f'SPair_{metric}', \
    pooled=False, base_colors=True, inc_filters='-B-16', suff='B-16', x_block=True, separate_legend=True, no_arch=True)
    best_block_table(data, names, metric, higher_better=True)

plot_spair(metric='pck@0.1')
plot_spair(metric='pck@0.0.5')
plot_spair(metric='pck@0.0.01')

def plot_retreival(dataset='roxford5k', metric='mapM'):
    print('Plotting retrieval metrics...')
    names = []
    data = []
    root_dir = 'all_results'
    for mod_id in sorted(os.listdir(root_dir)):
        if mod_id!='plots' and mod_id!='davis_dense' and mod_id!='plots_all':
            names.append(mod_id)
            mod_dir = os.path.join(root_dir, mod_id)
            knn_op = os.listdir(mod_dir)
            knn_op = [x for x in knn_op if dataset in x]
            knn_op.sort(key=lambda f: int(re.sub('\D', '', f)))
            temp_retrieval = []
            for op in knn_op:
                op_dir = os.path.join(mod_dir, op)
                with open(op_dir, 'rb') as f:
                    retrieval_metrics = pickle.load(f)
                temp_retrieval.append(retrieval_metrics[metric])
            data.append(np.array(temp_retrieval))
    pooled_blockwise_comparison_plot('all_results/plots', data, names, f'Retrieval_{dataset}_{metric}', \
    pooled=False, base_colors=True, inc_filters='-B-16', suff='B-16', x_block=True, separate_legend=True, no_arch=True)
    best_block_table(data, names, dataset+'_'+metric, higher_better=True)

plot_retreival('roxford5k', metric='mapM')
plot_retreival('rparis6k', metric='mapM')
plot_retreival('roxford5k', metric='mapH')
plot_retreival('rparis6k', metric='mapH')

def plot_davis(metric='mapM'):
    if metric=='jandfmean':
        ind = 0
    elif metric=='jmean':
        ind = 1
    elif metric=='fmean':
        ind = 4
    print('Plotting davis metrics...')
    names = []
    data = []
    root_dir = 'all_results/davis_dense'
    for mod_id in sorted(os.listdir(root_dir)):
        if mod_id!='plots' and mod_id!='plots_all':
            names.append(mod_id)
            mod_dir = os.path.join(root_dir, mod_id)
            knn_op = os.listdir(mod_dir)
            knn_op = [x for x in knn_op if 'davis' in x]
            knn_op.sort(key=lambda f: int(re.sub('\D', '', f)))
            temp_davis = []
            for op in knn_op:
                op_dir = os.path.join(mod_dir, op, 'global_results-val.csv')
                with open(op_dir, 'r') as f:
                    csv_reader = csv.reader(f, delimiter=',')
                    next(csv_reader, None)
                    for row in csv_reader:
                        jandfmean = float(row[ind])
                temp_davis.append(jandfmean)
            data.append(np.array(temp_davis))
    pooled_blockwise_comparison_plot('all_results/plots', data, names, f'DAVIS_{metric}', \
    pooled=False, base_colors=True, inc_filters='-B-16', suff='dense-B-16', x_block=True, separate_legend=True, no_arch=True)
    best_block_table(data, names, metric, higher_better=True)
plot_davis(metric = 'jandfmean')
plot_davis(metric = 'jmean')
plot_davis(metric = 'fmean')