"""
###########################################################################
Centered Kernal Analysis (CKA) for ViT features.

This code is adapted from https://github.com/AntixK/PyTorch-Model-Compare

Written by: Saksham Suri
###########################################################################
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from warnings import warn
from typing import List, Dict
import matplotlib.pyplot as plt
from .utils import add_colorbar
import torch.nn.functional as f
import numpy as np
import sys
sys.path.append('../meta_utils')
from meta_utils.plot_format import display_names

class CKA:
    def __init__(self,
                 model1: nn.Module,
                 model2: nn.Module,
                 runner1: None,
                 runner2: None,
                 model1_name: str = None,
                 model2_name: str = None,
                 model1_layers: List[str] = None,
                 model2_layers: List[str] = None,
                 device: str ='cpu'):
        """

        :param model1: (nn.Module) Neural Network 1
        :param model2: (nn.Module) Neural Network 2
        :param model1_name: (str) Name of model 1
        :param model2_name: (str) Name of model 2
        :param model1_layers: (List) List of layers to extract features from
        :param model2_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model1 = model1
        self.model2 = model2
        self.runner1 = runner1
        self.runner2 = runner2
        self.device = device

        self.model1_info = {}
        self.model2_info = {}

        if model1_name is None:
            self.model1_info['Name'] = model1.__repr__().split('(')[0]
        else:
            self.model1_info['Name'] = model1_name

        if model2_name is None:
            self.model2_info['Name'] = model2.__repr__().split('(')[0]
        else:
            self.model2_info['Name'] = model2_name

        if self.model1_info['Name'] == self.model2_info['Name']:
            warn(f"Both model have identical names - {self.model2_info['Name']}. " \
                 "It may cause confusion when interpreting the results. " \
                 "Consider giving unique names to the models :)")

        self.model1_info['Layers'] = []
        self.model2_info['Layers'] = []

        self.model1_features = {}
        self.model2_features = {}

        if len(list(model1.modules())) > 150 and model1_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model1_layers' parameter. Your CPU/GPU will thank you :)")

        self.model1_layers = model1_layers

        if len(list(model2.modules())) > 150 and model2_layers is None:
            warn("Model 2 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model2_layers' parameter. Your CPU/GPU will thank you :)")

        self.model2_layers = model2_layers
        
        self._insert_hooks()
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)

        self.model1.eval()
        self.model2.eval()

    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model1":
            self.model1_features[name] = out

        elif model == "model2":
            self.model2_features[name] = out

        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model1.named_modules():
            if self.model1_layers is not None:
                if name in self.model1_layers:
                    self.model1_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model1", name))
            else:
                self.model1_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model1", name))

        # Model 2
        for name, layer in self.model2.named_modules():
            if self.model2_layers is not None:
                if name in self.model2_layers:
                    self.model2_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model2", name))
            else:

                self.model2_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model2", name))

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def compare(self,
                dataloader1: DataLoader,
                dataloader2: DataLoader = None, mode = 'all') -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        """

        if dataloader2 is None:
            warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
            dataloader2 = dataloader1

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        self.hsic_matrix = torch.zeros(N, M, 3)

        num_batches = min(len(dataloader1), len(dataloader1))

        for (x1, *_), (x2, *_) in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):
            self.model1_features = {}
            self.model2_features = {}
            if('MAE' in self.model1_info['Name']):
                _ = self.runner1(x1.to(self.device), mask_ratio=0.0, no_shuffle=True)
            else:
                _ = self.runner1(x1.to(self.device))
            if('MAE' in self.model2_info['Name']):
                _ = self.runner2(x2.to(self.device), mask_ratio=0.0, no_shuffle=True)
            else:
                _ = self.runner2(x2.to(self.device))

            for i, (name1, feat1) in enumerate(self.model1_features.items()):
                if(type(feat1)==tuple):
                    feat1 = feat1[0]
                if('visual' in name1 and 'conv' not in name1):
                    X = feat1.permute(1,0,2) #remove permute for other models
                else:
                    X = feat1

                if(mode == 'cls'):
                    X = X[:,0,:].flatten(1).float()
                elif(mode == 'spat'):
                    X = X[:,1:,:].flatten(1).float()
                elif(mode == 'all'):
                    X = X.flatten(1).float()

                K = X @ X.t()
                K = K - K.min()
                K = K/K.max()
                K.fill_diagonal_(0.0)
                self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

                for j, (name2, feat2) in enumerate(self.model2_features.items()):
                    if(type(feat2)==tuple):
                        feat2 = feat2[0]
                    if('visual' in name2 and 'conv' not in name2):
                        Y = feat2.permute(1,0,2) #remove permute for other models
                    else:
                        Y = feat2

                    if(mode == 'cls'):
                        Y = Y[:,0,:].flatten(1).float()
                    elif(mode == 'spat'):
                        Y = Y[:,1:,:].flatten(1).float()
                    elif(mode == 'all'):
                        Y = Y.flatten(1).float()

                    L = Y @ Y.t()
                    L = L - L.min()
                    L = L/L.max()
                    L.fill_diagonal_(0)
                    assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                    self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                    self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches

        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                        self.hsic_matrix[:, :, 2].sqrt())

        assert not torch.isnan(self.hsic_matrix).any(), "HSIC computation resulted in NANs"
    
    def compare_residual(self,
                dataloader1: DataLoader,
                dataloader2: DataLoader = None) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        """

        if dataloader2 is None:
            warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
            dataloader2 = dataloader1

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

        
        num_batches = min(len(dataloader1), len(dataloader1))
        self.all_norm = []
        for i in range(len(self.model1_layers)):
            self.all_norm.append([])
        
        self.cls_norm = []
        for i in range(len(self.model1_layers)):
            self.cls_norm.append([])
        
        self.spat_norm = []
        for i in range(len(self.model1_layers)):
            self.spat_norm.append([])

        for (x1, *_), (x2, *_) in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):

            self.model1_features = {}
            self.model2_features = {}
            if('MAE' in self.model1_info['Name']):
                _ = self.runner1(x1.to(self.device), mask_ratio=0.0, no_shuffle=True)
            else:
                _ = self.runner1(x1.to(self.device))
            if('MAE' in self.model2_info['Name']):
                _ = self.runner2(x2.to(self.device), mask_ratio=0.0, no_shuffle=True)
            else:
                _ = self.runner2(x2.to(self.device))
            for ind in range(len(self.model1_features.keys())):
                all_feats_1 = self.model1_features[list(self.model1_features.keys())[ind]]
                all_feats_2 = self.model2_features[list(self.model2_features.keys())[ind]]
                if('visual' in list(self.model1_features.keys())[ind]):
                    all_feats_1 = all_feats_1.permute(1,0,2).float()
                    all_feats_2 = all_feats_2.permute(1,0,2).float()
                norm_all = torch.mean(all_feats_1.norm(dim=(1,2))/all_feats_2.norm(dim=(1,2)))
                norm_cls = torch.mean(all_feats_1[:,0,:].norm(dim=1)/all_feats_2[:,0,:].norm(dim=1))
                norm_spat = torch.mean(all_feats_1[:,1:,:].norm(dim=(1,2))/all_feats_2[:,1:,:].norm(dim=(1,2)))
                self.all_norm[ind].append(norm_all.item())
                self.cls_norm[ind].append(norm_cls.item())
                self.spat_norm[ind].append(norm_spat.item())

    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": self.model1_info['Name'],
            "model2_name": self.model2_info['Name'],
            "CKA": self.hsic_matrix.item(),
            "model1_layers": self.model1_info['Layers'],
            "model2_layers": self.model2_info['Layers'],
            "dataset1_name": self.model1_info['Dataset'],
            "dataset2_name": self.model2_info['Dataset'],

        }

    def plot_results(self,
                     save_path: str = None,
                     title: str = None):
        fig, ax = plt.subplots()
        im = ax.imshow(self.hsic_matrix, origin='lower', cmap='magma')
        if('B-16' in self.model1_info['Name'] and 'B-16' in self.model2_info['Name']):
            ax.set_xlabel(f"{display_names(self.model2_info['Name'], no_arch=True)}", fontsize=15)
            ax.set_ylabel(f"{display_names(self.model1_info['Name'], no_arch=True)}", fontsize=15)
        else:
            ax.set_xlabel(f"{display_names(self.model2_info['Name'])}", fontsize=15)
            ax.set_ylabel(f"{display_names(self.model1_info['Name'])}", fontsize=15)
            
        # if title is not None:
        #     ax.set_title(f"{title}", fontsize=18)
        # else:
        #     ax.set_title(f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=18)

        add_colorbar(im)
        plt.tight_layout()

        if save_path is not None:
            # save as pdf
            plt.savefig(save_path, bbox_inches='tight')
            save_path = save_path.replace('.png', '.pdf')
            plt.savefig(save_path, bbox_inches='tight')
            # save hsic matrix
            save_path = save_path.replace('.pdf', '.npy')
            np.save(save_path, self.hsic_matrix)

        plt.show()

    def plot_results_residual(self,
                     save_path: str = None,
                     title: str = None):

        fig, ax = plt.subplots()
        im = ax.imshow(self.hsic_matrix, origin='lower', cmap='magma')
        ax.set_xlabel('Normal Connection', fontsize=25)
        ax.set_ylabel('Skip Connection', fontsize=25)
        ax.set_title(f"{display_names(self.model1_info['Name'], no_arch=True)}", fontsize=28)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        add_colorbar(im)
        plt.tight_layout()

        if save_path is not None:
            # save as pdf
            save_path = save_path + '.png'
            plt.savefig(save_path, bbox_inches='tight')
            save_path = save_path.replace('.png', '.pdf')
            plt.savefig(save_path, bbox_inches='tight')
            # save the hsic matrix
            save_path = save_path.replace('.pdf', '.npy')
            np.save(save_path, self.hsic_matrix)
        plt.show()

class GetTok:
    def __init__(self,
                 model1: nn.Module,
                 runner1 = None,
                 model1_name: str = None,
                 model1_layers: List[str] = None,
                 mode: str = None,
                 device: str ='cpu'):
        self.model1 = model1
        self.runner1 = runner1
        self.device = device
        self.mode = mode

        self.model1_info = {}

        if model1_name is None:
            self.model1_info['Name'] = model1.__repr__().split('(')[0]
        else:
            self.model1_info['Name'] = model1_name

        self.model1_info['Layers'] = []
        self.model1_features = {}

        if len(list(model1.modules())) > 150 and model1_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model1_layers' parameter. Your CPU/GPU will thank you :)")

        self.model1_layers = model1_layers
        self._insert_hooks()
        self.model1 = self.model1.to(self.device)
        self.model1.eval()
        
    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model1":
            self.model1_features[name] = out

        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model1.named_modules():
            if self.model1_layers is not None:
                if name in self.model1_layers:
                    self.model1_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model1", name))
            else:
                self.model1_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model1", name))

    def dump_token(self, dataloader1: DataLoader) -> None:
        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        num_batches = min(len(dataloader1), len(dataloader1))
        self.feats = {}
        for layer in self.model1_info['Layers']:
            self.feats[layer] = []
        self.labels = []
        for (x1, lab) in tqdm(dataloader1, desc="| Comparing features |", total=num_batches):
            self.model1_features = {}
            if('MAE' in self.model1_info['Name']):
                _ = self.runner1(x1.to(self.device), mask_ratio=0.0, no_shuffle=True)
            else:
                _ = self.runner1(x1.to(self.device))
            self.labels.append(lab)
            for layer in self.model1_features.keys():
                if(self.mode=='cls'):
                    if('visual' in layer):
                        self.feats[layer].append(self.model1_features[layer].permute(1,0,2).cpu().numpy()[:,0,:])
                    else:
                        self.feats[layer].append(self.model1_features[layer].cpu().numpy()[:,0,:])
                elif(self.mode=='spat'):
                    if('visual' in layer):
                        self.feats[layer].append(self.model1_features[layer].permute(1,0,2).cpu().numpy()[:,1:,:].mean(1))
                    else:
                        self.feats[layer].append(self.model1_features[layer].cpu().numpy()[:,1:,:].mean(1))
                else:
                    print("ERROR: Please specify mode as cls or spat")
            
        for layer in self.model1_info['Layers']:
            self.feats[layer] = np.concatenate(self.feats[layer])
        self.labels = np.concatenate(self.labels)
