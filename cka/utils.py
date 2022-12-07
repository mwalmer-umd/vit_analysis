"""
###########################################################################
Feature analysis utilities

The purity code was adapted from https://stackoverflow.com/questions/34047540/python-clustering-purity-metric

Written by: Saksham Suri
###########################################################################
"""
from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt
import json
import torch
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score

def get_layers_cka(mod_id):
    # loads a pickle file with a list containing model layers we want to use for the CKA analysis
    with open('layer_jsons/cka_all/' + mod_id + '.json', 'r') as f:
        layers = json.load(f)
    return layers

def get_layers_residual(mod_id, ltype = 'attn_residual'):
    # loads a pickle file with a list containing model layers we want to use for the CKA analysis
    with open('layer_jsons/cka_residual/' + mod_id + '.json', 'r') as f:
        layers_ = json.load(f)
    layers = []
    for layer in layers_:
        if ltype in layer:
            layers.append(layer)
    return layers

def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)
    
def cluster_metric(k, dataset_feats, class_labels, silent=False):
    # set seed for kmeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(dataset_feats)
    class_labels = np.array(class_labels)
    nmi = normalized_mutual_info_score(class_labels, kmeans.labels_)
    ari = adjusted_rand_score(class_labels, kmeans.labels_)
    pur = purity_score(class_labels, kmeans.labels_)
    if not silent:
        print('NMI:  ', nmi)
        print('ARI:  ', ari)
        print('Purity:  ', pur)
    return nmi, ari, pur

def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def get_keep_list(mod_id):
    if ('CLIP' not in mod_id):
        if ('-S-' in mod_id or '-B-' in mod_id):
            return ['blocks.0', 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', \
                         'blocks.6', 'blocks.7', 'blocks.8', 'blocks.9', 'blocks.10', 'blocks.11']
        elif ('-L-' in mod_id):
            return ['blocks.0', 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', \
             'blocks.6', 'blocks.7', 'blocks.8', 'blocks.9', 'blocks.10', 'blocks.11', \
             'blocks.12','blocks.13','blocks.14', 'blocks.15','blocks.16','blocks.17',\
             'blocks.18','blocks.19','blocks.20','blocks.21','blocks.22','blocks.23']
        elif('-H-' in mod_id):
            return ['blocks.0', 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', \
             'blocks.6', 'blocks.7', 'blocks.8', 'blocks.9', 'blocks.10', 'blocks.11', \
             'blocks.12','blocks.13','blocks.14', 'blocks.15','blocks.16','blocks.17',\
             'blocks.18','blocks.19','blocks.20','blocks.21','blocks.22','blocks.23',\
             'blocks.24','blocks.25','blocks.26','blocks.27','blocks.28','blocks.29',\
             'blocks.30','blocks.31']
    elif ('CLIP' in mod_id):
        if('-B-' in mod_id):
            return ['visual.transformer.resblocks.0','visual.transformer.resblocks.1','visual.transformer.resblocks.2',\
                'visual.transformer.resblocks.3','visual.transformer.resblocks.4','visual.transformer.resblocks.5',\
                'visual.transformer.resblocks.6','visual.transformer.resblocks.7','visual.transformer.resblocks.8',\
                'visual.transformer.resblocks.9','visual.transformer.resblocks.10','visual.transformer.resblocks.11']
        else:
            return ['visual.transformer.resblocks.0','visual.transformer.resblocks.1','visual.transformer.resblocks.2',\
                'visual.transformer.resblocks.3','visual.transformer.resblocks.4','visual.transformer.resblocks.5',\
                'visual.transformer.resblocks.6','visual.transformer.resblocks.7','visual.transformer.resblocks.8',\
                'visual.transformer.resblocks.9','visual.transformer.resblocks.10','visual.transformer.resblocks.11',\
                'visual.transformer.resblocks.12','visual.transformer.resblocks.13','visual.transformer.resblocks.14',\
                'visual.transformer.resblocks.15','visual.transformer.resblocks.16','visual.transformer.resblocks.17',\
                'visual.transformer.resblocks.18','visual.transformer.resblocks.19','visual.transformer.resblocks.20',\
                'visual.transformer.resblocks.21','visual.transformer.resblocks.22','visual.transformer.resblocks.23']