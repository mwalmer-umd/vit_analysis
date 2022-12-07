"""
###########################################################################
Keypoint Correspondence evaluation with SPair-71k

Written by: Kamal Gupta
###########################################################################
"""
import os
import argparse
from glob import glob
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import pickle
from meta_utils.get_model_wrapper import get_model_wrapper
from meta_utils.result_cacher import read_results_cache, save_results_cache
from meta_utils.dense_extractor import dense_extractor
from analysis.attention_plots import meta_plot
from meta_utils.data_summary import best_block_table
from meta_utils.preproc import minimal_transform


#################### DATA LOADING ################
def resize(img, target_res, resize=True, to_pil=True):
    canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
    original_width, original_height = img.size
    if original_height <= original_width:
        if resize:
            img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
        width, height = img.size
        img = np.asarray(img)
        canvas[(width - height) // 2: (width + height) // 2] = img
    else:
        if resize:
            img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
        width, height = img.size
        img = np.asarray(img)
        canvas[:, (height - width) // 2: (height + width) // 2] = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas


def preprocess_kps_pad(kps, img_width, img_height, size):
    # Once an image has been pre-processed via border (or zero) padding,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the border-padded (or zero-padded) image.
    kps = kps.clone()
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = int((size - new_h) / 2)
        offset_x = 0
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = int((size - new_w) / 2)
        offset_y = 0
        kps[:, 0] += offset_x
    else:
        offset_x = 0
        offset_y = 0
    kps *= kps[:, 2:3]  # zero-out any non-visible key points
    return kps, offset_x, offset_y, scale


def load_spair_data(path, size=256, category='cat', split='test', subsample=None):
    pairs = sorted(glob(f'{path}/PairAnnotation/{split}/*_{category}.json'))
    assert len(pairs) > 0, '# of groundtruth image pairs must be > 0'
    if subsample is not None and subsample > 0:
        pairs = [pairs[ix] for ix in np.random.choice(len(pairs), subsample)]
    print(f'Number of SPairs for {category} = {len(pairs)}')
    files = []
    thresholds = []
    category_anno = list(glob(f'{path}/ImageAnnotation/{category}/*.json'))[0]
    with open(category_anno) as f:
        num_kps = len(json.load(f)['kps'])
    print(f'Number of SPair key points for {category} <= {num_kps}')
    kps = []
    blank_kps = torch.zeros(num_kps, 3)
    for pair in pairs:
        with open(pair) as f:
            data = json.load(f)
        assert category == data["category"]
        assert data["mirror"] == 0
        source_fn = f'{path}/JPEGImages/{category}/{data["src_imname"]}'
        target_fn = f'{path}/JPEGImages/{category}/{data["trg_imname"]}'
        source_bbox = np.asarray(data["src_bndbox"])
        target_bbox = np.asarray(data["trg_bndbox"])
        
        source_size = data["src_imsize"][:2]  # (W, H)
        target_size = data["trg_imsize"][:2]  # (W, H)

        kp_ixs = torch.tensor([int(id) for id in data["kps_ids"]]).view(-1, 1).repeat(1, 3)
        source_raw_kps = torch.cat([torch.tensor(data["src_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
        source_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=source_raw_kps)
        source_kps, src_x, src_y, src_scale = preprocess_kps_pad(source_kps, source_size[0], source_size[1], size)
        target_raw_kps = torch.cat([torch.tensor(data["trg_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
        target_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=target_raw_kps)
        target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(target_kps, target_size[0], target_size[1], size)

        # The source thresholds aren't actually used to evaluate PCK on SPair-71K, but for completeness
        # they are computed as well:
        thresholds_src = max(source_bbox[3] - source_bbox[1], source_bbox[2] - source_bbox[0]) * src_scale
        thresholds_trg = max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0]) * trg_scale
        thresholds.append(thresholds_src)
        thresholds.append(thresholds_trg)

        kps.append(source_kps)
        kps.append(target_kps)
        files.append(source_fn)
        files.append(target_fn)

    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]
    print(f'Final number of used key points: {kps.size(1)}')
    return files, kps, thresholds


#################### ANALYSIS METHODS ####################


def compute_pck(files, kps, thresholds, mod_wrap, imsize=224, patch=16, layer=9, meta_model='dino'):
    gt_correspondences = []
    pred_correspondences = []
    N = len(files) // 2
    num_patches = imsize // patch
    rescale_factor = imsize/num_patches
    transform = minimal_transform(meta_model)
    pbar = tqdm(total=N)

    thresholds_trg = []
    # thresholds_src = []

    for pair_idx in range(N):
        # Load image 1
        img1 = Image.open(files[2*pair_idx]).convert('RGB')
        img1 = resize(img1, imsize, resize=True, to_pil=True)
        img1_kps = kps[2*pair_idx]

        # Get patch index for the keypoints
        img1_y  = img1_kps[:, 1].round().numpy().astype(np.int32)
        img1_x = img1_kps[:, 0].numpy().astype(np.int32)
        img1_patch_idx = imsize * img1_y + img1_x

        # Load image 2
        img2 = Image.open(files[2*pair_idx+1]).convert('RGB')
        img2 = resize(img2, imsize, resize=True, to_pil=True)
        img2_kps = kps[2*pair_idx+1]

        img1 = transform(img1).unsqueeze(0).to(mod_wrap.device)
        fs1 = mod_wrap.get_activations(img1)[layer] # get features
        img2 = transform(img2).unsqueeze(0).to(mod_wrap.device)
        fs2 = mod_wrap.get_activations(img2)[layer] # get features

        # 1 x (1+num_patches*num_patches) x feat_dim => 1 x feat_dim x num_patches x num_patches
        fs1 = fs1[0, 1:].permute(1, 0).reshape(1, -1, num_patches, num_patches)
        fs1 = F.normalize(fs1, dim=1)
        fs2 = fs2[0][1:].permute(1, 0).reshape(1, -1, num_patches, num_patches)
        fs2 = F.normalize(fs2, dim=1)
        # 1 x feat_dim x num_patches x num_patches => 1 x feat_dim x imsize x imsize
        fs1 = F.interpolate(fs1, scale_factor=rescale_factor, mode='bilinear')
        fs2 = F.interpolate(fs2, scale_factor=rescale_factor, mode='bilinear')
        # 1 x feat_dim x imsize x imsize => imsize*imsize x feat_dim
        fs1 = fs1[0].permute(1,2,0).reshape(imsize*imsize, -1)
        fs2 = fs2[0].permute(1,2,0).reshape(imsize*imsize, -1)

        # Get mutual visibility
        vis = img1_kps[:, 2] * img2_kps[:, 2] > 0

        # Get similarity matrix
        sim_1_to_2 = fs1[img1_patch_idx[vis]] @ fs2.permute(1, 0)
        # Get nearest neighors
        nn_1_to_2 = torch.argmax(sim_1_to_2, dim=-1)
        nn_y, nn_x = nn_1_to_2 // imsize, nn_1_to_2 % imsize
        kps_1_to_2 = torch.stack([nn_x, nn_y]).permute(1, 0)

        gt_correspondences.append(img2_kps[vis][:, :2])
        pred_correspondences.append(kps_1_to_2)
        
        # gather thresholds
        n_kp = kps_1_to_2.shape[0]
        threshs = torch.ones(n_kp, dtype=torch.float32) * thresholds[2*pair_idx+1]
        thresholds_trg.append(threshs)

        pbar.update(1)

    gt_correspondences = torch.cat(gt_correspondences, dim=0).cpu()
    pred_correspondences = torch.cat(pred_correspondences, dim=0).cpu()
    thresholds_trg = torch.cat(thresholds_trg, dim=0).cpu()

    alpha = torch.tensor([0.1, 0.05, 0.01])
    correct = torch.zeros(len(alpha))

    err = (pred_correspondences - gt_correspondences).norm(dim=-1)
    err = err.unsqueeze(0).repeat(len(alpha), 1)
    
    thresholds = thresholds_trg.unsqueeze(0).repeat(len(alpha), 1)
    thresholds *= alpha.unsqueeze(-1)

    correct = err < thresholds
    correct = correct.sum(dim=-1) / len(gt_correspondences)

    alpha2pck = zip(alpha.tolist(), correct.tolist())
    print(' | '.join([f'PCK-Transfer@{alpha:.2f}: {pck_alpha * 100:.2f}%'
                      for alpha, pck_alpha in alpha2pck]))
    return correct


#################### ANALYSIS ENGINE ####################


def load_or_run_analysis(args):
    # prep model wrapper
    # if you want to extract the attention maps use:
    # mod_wrap = get_model_wrapper(args.meta_model, args.arch, args.patch, args.imsize, extract_mode='attn', blk_sel='all')
    # if you want to extract the feature maps use:
    mod_wrap = get_model_wrapper(args.meta_model, args.arch, args.patch, args.imsize, extract_mode='feat', blk_sel='all')
    
    # prep names for analysis methods
    # each analysis method is given a unique name to aid in cache and plot saving
    analysis_methods = ['pck@0.1', 'pck@0.0.5', 'pck@0.0.01']

    # check cache
    if not (args.nocache or args.overcache):
        results, found, not_found = read_results_cache(mod_wrap.mod_id, analysis_methods)
        if len(results) == len(analysis_methods):
            return results, analysis_methods

    # load model
    mod_wrap.load()

    # set seed
    np.random.seed(args.seed)
    # prep dataset
    all_results = {a_m: [] for a_m in analysis_methods}
    dataset = None
    categories = os.listdir(os.path.join(args.dataroot, 'ImageAnnotation'))
    assert len(categories) > 0, '# of categories must be > 0'
    for cat in categories:
        files, kps, thresholds = load_spair_data(args.dataroot, size=args.imsize, category=cat,
                                     subsample=args.pairs_per_category)
        # run metrics
        print(f'Running Metrics on {cat}')
        correct = compute_pck(files, kps, thresholds, mod_wrap, imsize=args.imsize, patch=args.patch, layer=int(args.blk), meta_model=args.meta_model)
        for i, k in enumerate(analysis_methods):
            all_results[k].append(correct[i].item())

    # stack results
    results = []
    for a_m in analysis_methods:
        print(f'{a_m}:{np.mean(all_results[a_m])*100:.2f}%')
    
    # dump results
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, mod_wrap.mod_id), exist_ok=True)
    with open(os.path.join(args.output_dir, mod_wrap.mod_id, f'spair_kp_blk{args.blk}.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    
    return results, analysis_methods



#################### MAIN ####################



def main(args):
    load_or_run_analysis(args)




def parse_args():
    parser = argparse.ArgumentParser('Generic analysis template')
    ######### GENERAL
    parser.add_argument('--overcache', action='store_true', help='disable cache reading but over-write cache when finished')
    parser.add_argument('--nocache', action='store_true', help='fully disable reading and writing of cache files (overrides --overcache)')
    parser.add_argument('--output_dir', default='all_results', help='dir to save metric plots to')   
    ######### MODEL
    parser.add_argument('--meta_model', default='dino', choices=['dino', 'clip', 'mae', 'timm', 'moco', 'beit', 'random'], help='style of model to load')
    parser.add_argument('--arch', default='B', type=str, choices=['T', 'S', 'B', 'L', 'H'], help='size of vit to load')
    parser.add_argument('--patch', default=16, type=int, help='vit patch size')
    parser.add_argument('--imsize', default=224, type=int, help='image resize size')
    parser.add_argument('--blk', default='all', type=str, help='which block to extract features from (first, q1, middle, q3, last, <INT>) default: last')
    ######### DATASET
    parser.add_argument('--dataroot', default='data/SPair-71k')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--pairs_per_category', type=int, default=20, help='how many images pairs to sample from each category')
    ######### METRICS
    #########
    args = parser.parse_args()
    return args
    


if __name__ == '__main__':
    args = parse_args()
    main(args)