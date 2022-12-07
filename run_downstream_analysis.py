"""
###########################################################################
Downstream task analysis on three different types of task: knn
classification, image retrieval, and DAVIS video segmentation.

This code is built on code from  https://github.com/facebookresearch/dino.
Original copyright notice shown below.

Written by: Saksham Suri
###########################################################################
"""
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import time
import colorsys
import shutil
import json
import torch
import numpy as np
from PIL import Image, ImageFile, ImageColor
from torch.utils.data import DataLoader
from tqdm import tqdm
from meta_utils.get_model_wrapper import get_model_wrapper
import matplotlib.pyplot as plt
from torch import nn
from torchvision.datasets import ImageFolder
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import eval_utils
from eval_utils import OxfordParisDataset, knn_classifier
import pickle
from eval_video_segmentation import eval_video_tracking_davis, read_seg, read_frame_list
from urllib.request import urlopen
from torchvision import transforms
from meta_utils.dense_extractor import dense_extractor


@torch.no_grad()
def extract_features(args):
    mod_wrap, mod_id, train_dataloader, test_dataloader, gnd = load_or_run_analysis(args)
    print(mod_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features_train = []
    labels_train = []
    for samples, labs in tqdm(train_dataloader):
        samples = samples.to(device)
        feats = mod_wrap.get_activations(samples)
        feats = feats[0][:,0,:] # take only cls token
        features_train.append(feats.cpu())
        labels_train.append(labs)
    features_train = torch.cat(features_train, dim=0)
    labels_train = torch.cat(labels_train, dim=0)

    features_test = []
    labels_test = []
    for samples, labs in tqdm(test_dataloader):
        samples = samples.to(device)
        feats = mod_wrap.get_activations(samples)
        feats = feats[0][:,0,:] # take only cls token
        features_test.append(feats.cpu())
        labels_test.append(labs)
    features_test = torch.cat(features_test, dim=0)
    labels_test = torch.cat(labels_test, dim=0)
    return (features_train, labels_train, features_test, labels_test, gnd, mod_id)


#################### MODEL RUNNING ####################


def load_or_run_analysis(args):
    mod_wrap = get_model_wrapper(args.meta_model, args.arch, args.patch, args.imsize, 'feat', args.blk)
    mod_id = mod_wrap.mod_id

    if(args.run_knn and os.path.exists(os.path.join(args.output_dir, mod_id, f'{args.dataset}_knn_blk{args.blk}.pkl'))):
        print('knn already exists')
        exit()
    if(args.run_retrieval and os.path.exists(os.path.join(args.output_dir, mod_id, f'{args.dataset}_retrieval_blk{args.blk}.pkl'))):
        print('retrieval already exists')
        exit()
    # load model, prep dataloader
    print('loading model...')
    mod_wrap.load()
    if(args.dataset=='imagenet'):
        train_dataset = ImageFolder(root=args.dataroot+'train', transform=mod_wrap.transform)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch, num_workers=16, shuffle=True, pin_memory=False)
        test_dataset = ImageFolder(root=args.dataroot+'val', transform=mod_wrap.transform)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch, num_workers=16, shuffle=True, pin_memory=False)
        gnd = 0
    elif (args.dataset=='roxford5k' or args.dataset=='rparis6k'):
        dataset_train = OxfordParisDataset(args.dataroot, args.dataset, split="train", transform=mod_wrap.transform, imsize=args.imsize)
        train_dataloader = DataLoader(dataset_train, batch_size=args.batch, shuffle=False, pin_memory=True)
        dataset_test = OxfordParisDataset(args.dataroot, args.dataset, split="query", transform=mod_wrap.transform, imsize=args.imsize)
        test_dataloader = DataLoader(dataset_test, batch_size=args.batch, shuffle=False, pin_memory=True)
        gnd = dataset_train.cfg['gnd']
    elif (args.dataset=='davis'):
        if args.dense:
            mod_wrap = dense_extractor(mod_wrap, batch_limit=args.batch, cpu_assembly=args.cpu_assembly)
            return mod_wrap, mod_wrap.mod_id
        return (mod_wrap, mod_id)
    return (mod_wrap, mod_id, train_dataloader, test_dataloader, gnd)


@torch.no_grad()
def run_knn(args):
    train_features, train_labels, test_features, test_labels, _, mod_id = extract_features(args)
    train_features = nn.functional.normalize(train_features.float(), dim=1, p=2)
    test_features = nn.functional.normalize(test_features.float(), dim=1, p=2)

    print("Features are ready!\nStart the k-NN classification.")
    results_dict = {}
    for k in args.nb_knn:
        top1, top5 = knn_classifier(train_features, train_labels,
            test_features, test_labels, k, args.temperature)
        results_dict[k] = {'top1': top1, 'top5': top5}
        print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")

    # save results dict
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, mod_id), exist_ok=True)
    with open(os.path.join(args.output_dir, mod_id, f'{args.dataset}_knn_blk{args.blk}.pkl'), 'wb') as f:
        pickle.dump(results_dict, f)
    return


def run_video_segmentation(args):
    mod_wrap, mod_id = load_or_run_analysis(args)
    if not args.dense:
        new_transform = transforms.Compose([
            transforms.Resize((args.imsize, args.imsize), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            mod_wrap.transform.transforms[-1]])
        mod_wrap.transform = new_transform
    color_palette = []
    for line in urlopen("https://raw.githubusercontent.com/Liusifei/UVC/master/libs/data/palette.txt"):
        color_palette.append([int(i) for i in line.decode("utf-8").split('\n')[0].split(" ")])
    color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1,3)
    video_list = open(os.path.join(args.dataroot, "ImageSets/2017/val.txt")).readlines()
    for i, video_name in enumerate(video_list):
        video_name = video_name.strip()
        print(f'[{i}/{len(video_list)}] Begin to segmentate video {video_name}.')
        video_dir = os.path.join(args.dataroot, "JPEGImages/480p/", video_name)
        frame_list = read_frame_list(video_dir)
        seg_path = frame_list[0].replace("JPEGImages", "Annotations").replace("jpg", "png")
        if(args.dense):
            first_seg, seg_ori = read_seg(seg_path, args.patch, scale_size=[448])
        else:
            first_seg, seg_ori = read_seg(seg_path, args.patch, scale_size=[args.imsize, args.imsize])
        eval_video_tracking_davis(args, mod_wrap, frame_list, video_dir, first_seg, seg_ori, color_palette)


@torch.no_grad()
def run_retrieval(args):
    train_features, train_indices, query_features, query_indices, gnd, mod_id = extract_features(args)

    train_features = nn.functional.normalize(train_features.float(), dim=1, p=2)
    query_features = nn.functional.normalize(query_features.float(), dim=1, p=2)
    ############################################################################
    # Step 2: similarity
    sim = torch.mm(train_features, query_features.T)
    ranks = torch.argsort(-sim, dim=0).cpu().numpy()
    ############################################################################
    # Step 3: evaluate
    # gnd = dataset_train.cfg['gnd']
    # evaluate ranks
    ks = [1, 5, 10]
    # search for easy & hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk']])
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = eval_utils.compute_map(ranks, gnd_t, ks)
    # search for hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = eval_utils.compute_map(ranks, gnd_t, ks)
    print('>> {}: mAP M: {}, H: {}'.format(args.dataset, np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
    print('>> {}: mP@k{} M: {}, H: {}'.format(args.dataset, np.array(ks), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))
    # save results dict
    results_dict = {}
    results_dict['mapM'] = mapM
    results_dict['apsM'] = apsM
    results_dict['mprM'] = mprM
    results_dict['prsM'] = prsM
    results_dict['mapH'] = mapH
    results_dict['apsH'] = apsH
    results_dict['mprH'] = mprH
    results_dict['prsH'] = prsH
    results_dict['ks'] = ks
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, mod_id), exist_ok=True)
    with open(os.path.join(args.output_dir, mod_id, f'{args.dataset}_retrieval_blk{args.blk}.pkl'), 'wb') as f:
        pickle.dump(results_dict, f)


#################### MAIN ####################


def main():
    args = parse_args()
    if(args.run_knn):
        run_knn(args)
    if(args.run_retrieval):
        run_retrieval(args)
    if(args.run_video_segmentation):
        run_video_segmentation(args)

 
def parse_args():
    parser = argparse.ArgumentParser('Run feature analysis')
    ######### GENERAL
    parser.add_argument('--output_dir', default='all_results', help='dir to save metric plots to')
    ######### DATASET
    parser.add_argument('--dataset', default='roxford5k', type=str, choices=['roxford5k', 'rparis6k', 'imagenet', 'davis'])
    parser.add_argument('--dataroot', default='data/imagenet/')
    parser.add_argument('--batch', type=int, default=8)
    ######### MODEL
    parser.add_argument('--meta_model', default='dino', choices=['dino', 'clip', 'mae', 'timm', 'moco', 'beit', 'random'], help='style of model to load')
    parser.add_argument('--arch', default='B', type=str, choices=['T', 'S', 'B', 'L', 'H'], help='size of vit to load')
    parser.add_argument('--patch', default=16, type=int, help='vit patch size')
    parser.add_argument('--imsize', default=224, type=int, help='image resize size')
    parser.add_argument('--blk', default='last', type=str, help='which block to extract features from (first, q1, middle, q3, last, <INT>) default: last')
    ######### KNN
    parser.add_argument('--run_knn', action='store_true', help='run KNN')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    ######### RETRIEVAL
    parser.add_argument('--run_retrieval', action='store_true', help='run KNN')
    parser.add_argument('--dense', action='store_true', help='run KNN')
    parser.add_argument('--cpu_assembly', action='store_true', help='for use with --dense, gather dense features on cpu instead of gpu to avoid running out of memory')
    ######### VIDEO SEGMENTATION
    parser.add_argument('--run_video_segmentation', action='store_true', help='run video object segmentation')
    parser.add_argument("--n_last_frames", type=int, default=7, help="number of preceeding frames")
    parser.add_argument("--size_mask_neighborhood", default=12, type=int,
        help="We restrict the set of source nodes considered to a spatial neighborhood of the query node")
    parser.add_argument("--topk", type=int, default=5, help="accumulate label from top k neighbors")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()