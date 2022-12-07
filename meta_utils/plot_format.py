# utilities for uniform plot and line formatting
import colorsys
import matplotlib



DISPLAY_NAMES = {
    'BEIT' : 'BEiT',
    'CLIP' : 'CLIP',
    'DINO' : 'DINO',
    'MAE' : 'MAE',
    'TIMM' : 'FS',
    'MOCO' : 'MoCo',
    'RANDOM' : 'Random',
}
# helper to substitute final names for figures, specfically for model ids
# if no_arch==True, will hide the arch name
# if no_sup==True, will hide the supervision method
def display_names(name, no_arch=False, no_sup=False, no_dense=False):
    assert not (no_arch and no_sup)
    # update supervision name
    if name in DISPLAY_NAMES:
        return DISPLAY_NAMES[name]
    sup = name.split('-')[0]
    for base in DISPLAY_NAMES.keys():
        if base in name:
            name = name.replace(base, DISPLAY_NAMES[base])
            sup = DISPLAY_NAMES[base]
            break
    # remove extra elements
    name = name.replace('-ViT-',' ')
    name = name.replace('-224','')
    name = name.replace('-','/')
    if no_arch:
        name = name.split(' ')[0]
    if no_sup:
        name = name.split(' ')[-1]
    if no_dense:
        name = name.replace('/dense','')
    return name



METRIC_DISPLAY_NAMES = {
    'avg-att-dist' : ('Attention Distance', 'Distance'),
    'cls_att_align_iou_[pin]' : ('CLS Attention Saliency', 'IoU'),
    'spc_att_align_iou_[pin]' : ('Spat. Attention Saliency', 'IoU'),
    'seg-feat-pur_[coco]' : ('Object Cluster Purity', 'Purity'),
    'seg-feat-pur_[pin]' : ('Part Cluster Purity', 'Purity'),
    'seg-feat-nmi_[coco]' : ('Object Cluster NMI', 'NMI'),
    'seg-feat-nmi_[pin]' : ('Part Cluster NMI', 'NMI'),
    'seg-feat-ari_[coco]' : ('Object Cluster ARI', 'ARI'),
    'seg-feat-ari_[pin]' : ('Part Cluster ARI', 'ARI'),
    'DAVIS_jandfmean': ('Video Object Segmentation', 'J & F Mean'),
    'DAVIS_jmean': ('Video Object Segmentation', 'J Mean'),
    'DAVIS_fmean': ('Video Object Segmentation', 'F Mean'),
    'Retrieval_roxford5k_mapM': ('Image Retrieval', 'mAP'),
    'Retrieval_rparis6k_mapM': ('Image Retrieval', 'mAP'),
    'Retrieval_roxford5k_mapH': ('Image Retrieval', 'mAP'),
    'Retrieval_rparis6k_mapH': ('Image Retrieval', 'mAP'),
    'SPair_pck@0.1': ('Keypoint Correspondence', 'PCK@0.1'),
    'SPair_pck@0.0.5': ('Keypoint Correspondence', 'PCK@0.05'),
    'SPair_pck@0.0.01': ('Keypoint Correspondence', 'PCK@0.001'),
    'KNN_top1_Acc': ('k-NN Classification', 'Top-1 Accuracy'),
    'KNN_top5_Acc': ('k-NN Classification', 'Top-5 Accuracy'),
    'NMI': ('Image Clustering (CLS)', 'NMI'),
    'ARI': ('Image Clustering (CLS)', 'ARI'),
    'Purity': ('Image Clustering (CLS)', 'Purity'),
    'NMI_Spat': ('Image Clustering (Spat.)', 'NMI'),
    'ARI_Spat': ('Image Clustering (Spat.)', 'ARI'),
    'Purity_Spat': ('Image Clustering (Spat.)', 'Purity'),
}
PERTYPE_METRIC_DISPLAY_NAMES = {
    'cls_att_align_iou_[pin]' : ('CLS Attention Saliency (PartImageNet)', 'IoU'),
    'spc_att_align_iou_[pin]' : ('Spatial Attention Saliency (PartImageNet)', 'IoU'),
    'cls_att_align_iou_[coco]' : ('CLS Attention Saliency (COCO)', 'IoU'),
    'spc_att_align_iou_[coco]' : ('Spatial Attention Saliency (COCO)', 'IoU'),
    'DAVIS_jandfmean': ('DAVIS Video Object Segmentation', 'J & F Mean'),
    'DAVIS_jmean': ('DAVIS Video Object Segmentation', 'J Mean'),
    'DAVIS_fmean': ('DAVIS Video Object Segmentation', 'F Mean'),
    'Retrieval_roxford5k_mapM': ('ROxford5k Image Retrieval (Split-M)', 'mAP'),
    'Retrieval_rparis6k_mapM': ('RParis6k Image Retrieval (Split-M)', 'mAP'),
    'Retrieval_roxford5k_mapH': ('ROxford5k Image Retrieval (Split-H)', 'mAP'),
    'Retrieval_rparis6k_mapH': ('RParis6k Image Retrieval (Split-H)', 'mAP'),
    'SPair_pck@0.1': ('SPair-71k Keypoint Correspondence', 'PCK@0.1'),
    'SPair_pck@0.0.5': ('SPair-71k Keypoint Correspondence', 'PCK@0.05'),
    'SPair_pck@0.0.01': ('SPair-71k Keypoint Correspondence', 'PCK@0.001'),
    'KNN_top1_Acc': ('ImageNet k-NN Classification', 'Top-1 Accuracy'),
    'KNN_top5_Acc': ('ImageNet k-NN Classification', 'Top-5 Accuracy'),
}
# helper to substitute final names for figures. Given an analysis method
# name, will return both: a plot title and a y-axis label for said metric
# if the analysis method does not appear in the dict above, will return
# the same method name twice.
# UPDATE, when pertype is true, will check PERTYPE_METRIC_DISPLAY_NAMES,
# potentially overriding the settings in METRIC_DISPLAY_NAMES
def metric_display_names(a_m, pertype=False):
    if pertype and a_m in PERTYPE_METRIC_DISPLAY_NAMES:
        return PERTYPE_METRIC_DISPLAY_NAMES[a_m]
    if a_m in METRIC_DISPLAY_NAMES:
        return METRIC_DISPLAY_NAMES[a_m]
    return a_m, a_m



# settings for general plots:
AXIS_MARKER_INFO = {
    'avg-att-dist' : ([0.2, 0.3, 0.4, 0.5], 0.18, 0.58),
    # 'cls-merge_att_align_iou_[pin]' : ([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], -0.02, 0.55),
    # 'spc-merge_att_align_iou_[pin]' : ([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], -0.02, 0.55),
    'cls_att_align_iou_[pin]' : ([0.1, 0.2, 0.3, 0.4, 0.5], 0.08, 0.57),
    'spc_att_align_iou_[pin]' : ([0.1, 0.2, 0.3, 0.4, 0.5], 0.08, 0.57),
    'b16_blocks' : ([1, 3, 5, 7, 9, 11], 0.8, 12.2),
    'seg-feat-pur_[coco]' : ([0.3, 0.4, 0.5, 0.6, 0.7], 0.3, 0.67),
    # 'seg-feat-pur_[pin]' : ([0.2, 0.3, 0.4, 0.5, 0.6], 0.15, 0.63),
    'seg-feat-pur_[pin]' : ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 0.05, 0.63),
    # 'DAVIS_jandfmean':([0.3, 0.4, 0.5, 0.6], 0.26, 0.62),
    # 'DAVIS_jmean': ([0.3, 0.4, 0.5, 0.6], 0.28, 0.61),
    # 'DAVIS_fmean': ([0.3, 0.4, 0.5, 0.6], 0.25, 0.65),
    # 'Retrieval_roxford5k_mapM': ([0.2, 0.3, 0.4, 0.5], 0.18, 0.58),
    # 'Retrieval_rparis6k_mapM': ([0.2, 0.3, 0.4, 0.5], 0.18, 0.58),
    # 'Retreival_roxford5k_mapH': ([0.2, 0.3, 0.4, 0.5], 0.18, 0.58),
    # 'Retreival_rparis6k_mapH': ([0.2, 0.3, 0.4, 0.5], 0.18, 0.58),
    # 'SPair_pck@0.1':([0.2, 0.3, 0.4, 0.5], 0.18, 0.58),
    # 'SPair_pck@0.0.5': ([0.2, 0.3, 0.4, 0.5], 0.18, 0.58),
    # 'SPair_pck@0.0.01': ([0.2, 0.3, 0.4, 0.5], 0.18, 0.58),
    'KNN_top1_Acc': ([0, 20, 40, 60, 80], -1, 90),
    'KNN_top5_Acc': ([0, 20, 40, 60, 80, 100], -1, 100),
}
# settings for pertype plots:
AXIS_MARKER_INFO_PERTYPE = {
    'avg-att-dist' : ([0.2, 0.3, 0.4, 0.5, 0.6], 0.17, 0.65),
    'cls_att_align_iou_[pin]' : ([0.0, 0.2, 0.4, 0.6], 0.0, 0.6),
    'spc_att_align_iou_[pin]' : ([0.0, 0.2, 0.4, 0.6], 0.0, 0.6),
    'cls_att_align_iou_[coco]' : ([0.0, 0.2, 0.4, 0.6], 0.0, 0.6),
    'spc_att_align_iou_[coco]' : ([0.0, 0.2, 0.4, 0.6], 0.0, 0.6),
    'seg-feat-pur_[coco]' : ([0.3, 0.4, 0.5, 0.6, 0.7], 0.3, 0.71),
    'seg-feat-nmi_[coco]' : ([0.0, 0.2, 0.4, 0.6], 0.0, 0.65),
    'seg-feat-ari_[coco]' : ([0.0, 0.1, 0.2], -0.01, 0.2),
    'seg-feat-pur_[pin]' : ([0.1, 0.3, 0.5, 0.7], 0.05, 0.7),
    'seg-feat-nmi_[pin]' : ([0.0, 0.2, 0.4, 0.6], 0.0, 0.7),
    'seg-feat-ari_[pin]' : ([0.0, 0.1, 0.2, 0.3], -0.01, 0.35),
    'KNN_top1_Acc': ([0, 20, 40, 60, 80], -3, 90),
    'KNN_top5_Acc': ([0, 20, 40, 60, 80, 100], -3, 100),
    'SPair_pck@0.1':([5,15,25,35], 0, 40),
    'SPair_pck@0.0.5': ([5, 10, 15, 20], -0.03, 23),
    'SPair_pck@0.0.01': ([0.4, 0.8, 1.2, 1.6], -0.02, 1.8),
    'Retrieval_roxford5k_mapM': ([0.1, 0.2, 0.3, 0.4], 0, 0.48),
    'Retrieval_roxford5k_mapH': ([0.04, 0.08, 0.12, 0.16], 0, 0.2),
    'Retrieval_rparis6k_mapH': ([0.1, 0.2, 0.3, 0.4, 0.5], 0, 0.55),
    'Retrieval_rparis6k_mapM': ([0.1, 0.3, 0.5,0.7], 0, 0.8),
    'DAVIS_jandfmean':([0.1, 0.3, 0.5, 0.7], 0, 0.8),
    'DAVIS_jmean': ([0.1, 0.3, 0.5, 0.7], 0, 0.72),
    'DAVIS_fmean': ([0.1, 0.3, 0.5, 0.7], 0, 0.8),
    'NMI': ([0.1, 0.3, 0.5, 0.7, 0.9], 0, 1.0),
    'ARI': ([0.1, 0.3, 0.5, 0.7, 0.9], 0, 0.95),
    'Purity': ([0.1, 0.3, 0.5, 0.7, 0.9], 0, 0.98),
    'NMI_Spat': ([0.1, 0.3, 0.5, 0.7, 0.9], 0, 0.98),
    'ARI_Spat': ([0.1, 0.3, 0.5, 0.7, 0.9], 0, 0.95),
    'Purity_Spat': ([0.1, 0.3, 0.5, 0.7, 0.9], 0, 0.98),

}
# return information on axis markers, including tick positions
# min and max. If not specified, returns none
def axis_marker_info(a_m, pertype=False):
    if pertype:
        if a_m in AXIS_MARKER_INFO_PERTYPE:
            return AXIS_MARKER_INFO_PERTYPE[a_m]
    else:
        if a_m in AXIS_MARKER_INFO:
            return AXIS_MARKER_INFO[a_m]
    return None



FONT_DISPLAY_SCALE = {
    'avg-att-dist' : 1.438,
    'cls_att_align_iou_[coco]' : 1.438,
    'spc_att_align_iou_[coco]' : 1.438,
    'cls_att_align_iou_[pin]' : 1.438,
    'spc_att_align_iou_[pin]' : 1.438,
    'seg-feat-pur_[coco]' : 1.438,
    'seg-feat-pur_[pin]' : 1.438,
    'DAVIS_jandfmean': [1.438, 1.23],
    'DAVIS_jmean': [1.438, 1.23],
    'DAVIS_fmean': [1.438, 1.23],
    'Retrieval_roxford5k_mapM':  1.438,
    'Retrieval_rparis6k_mapM':  1.438,
    'Retrieval_roxford5k_mapH':  1.438,
    'Retrieval_rparis6k_mapH':  1.438,
    'SPair_pck@0.1':  [1.438, 1.23],
    'SPair_pck@0.0.5':  [1.438, 1.23],
    'SPair_pck@0.0.01':  [1.438, 1.23],
    'KNN_top1_Acc':  1.438,
    'KNN_top5_Acc':  1.438,
    'NMI': 1.438,
    'ARI': 1.438,
    'Purity': 1.438,
    'NMI_Spat': 1.438,
    'ARI_Spat': 1.438,
    'Purity_Spat': 1.438,
}
# look up font scaling info for plots (depending on display size)
def font_scale_info(a_m):
    if a_m not in FONT_DISPLAY_SCALE:
        return 1.438
    return FONT_DISPLAY_SCALE[a_m]



# list all analysis methods that should be plotted on bottom:
PERTYPE_BOTTOM_PLOTS = ['avg-att-dist', 'spc_att_align_iou_[pin]', 'seg-feat-ami_[pin]', 'seg-feat-ami_[coco]', 'KNN_top5_Acc', 'ARI', 'ARI_Spat', 'SPair_pck@0.0.01', 'Retrieval_rparis6k_mapH', 'DAVIS_jandfmean']
# for per-type blockwise comparison plots, plots can be
# "bottom" or "non-bottom". Only for bottom plots will there
# be x-axis labels and a legend, to make it easy to stack
# multiple plots
def is_bottom_plot(a_m):
    return (a_m in PERTYPE_BOTTOM_PLOTS)



##################################################



# SIMPLE_COLORS = {
#     'BEIT-ViT-B-16-224': 'aqua',
#     'BEIT-ViT-L-16-224': 'darkturquoise',
#     'CLIP-ViT-B-32-224': 'green',
#     'CLIP-ViT-B-16-224': 'limegreen',
#     'CLIP-ViT-L-14-224': 'lime',
#     'DINO-ViT-S-16-224': 'darkred',
#     'DINO-ViT-S-8-224':  'red',
#     'DINO-ViT-B-16-224': 'darkorange',
#     'DINO-ViT-B-8-224':  'orange',
#     'MAE-ViT-B-16-224':  'mediumvioletred',
#     'MAE-ViT-L-16-224':  'deeppink',
#     'MAE-ViT-H-14-224':  'hotpink',
#     'MOCO-ViT-S-16-224': 'magenta',
#     'MOCO-ViT-B-16-224': 'orchid',
#     'TIMM-ViT-S-32-224': 'darkblue',
#     'TIMM-ViT-S-16-224': 'mediumblue',
#     'TIMM-ViT-B-32-224': 'blue',
#     'TIMM-ViT-B-16-224': 'mediumslateblue',
#     'TIMM-ViT-B-8-224':  'blueviolet',
#     'TIMM-ViT-L-16-224': 'indigo',
#     'RANDOM-ViT-B-16-224': 'magenta',
# }
SIMPLE_COLORS = {
    'TIMM-ViT-S-32-224'   : 'lightcoral',
    'TIMM-ViT-S-16-224'   : 'lightcoral',
    'TIMM-ViT-B-32-224'   : 'darkred',
    'TIMM-ViT-B-16-224'   : 'darkred',
    'TIMM-ViT-B-8-224'    : 'darkred',
    'TIMM-ViT-L-16-224'   : 'black',
    'CLIP-ViT-B-32-224'   : 'red',
    'CLIP-ViT-B-16-224'   : 'red',
    'CLIP-ViT-L-14-224'   : 'maroon',
    'DINO-ViT-S-16-224'   : 'yellowgreen',
    'DINO-ViT-S-8-224'    : 'yellowgreen',
    'DINO-ViT-B-16-224'   : 'darkgreen',
    'DINO-ViT-B-8-224'    : 'darkgreen',
    'MOCO-ViT-S-16-224'   : 'lime',
    'MOCO-ViT-B-16-224'   : 'limegreen',
    'MAE-ViT-B-16-224'    : 'blue',
    'MAE-ViT-L-16-224'    : 'mediumblue',
    'MAE-ViT-H-14-224'    : 'darkblue',
    'BEIT-ViT-B-16-224'   : 'darkturquoise',
    'BEIT-ViT-L-16-224'   : 'darkcyan',
    'RANDOM-ViT-B-16-224' : 'magenta',
}
SIMPLE_LINESTYLES = {
    '32' : 'dashed',
    '16' : 'solid',
    '14' : 'dashdot',
    '8'  : 'dotted',
}
# POOLED_METHODS = {
#     'BEIT': 'darkturquoise',
#     'CLIP': 'limegreen',
#     'DINO': 'red',
#     'MAE': 'darkorange',
#     'TIMM': 'blue',
#     'MOCO': 'magenta'
# }
POOLED_METHODS = {
    'TIMM': 'darkred',
    'CLIP': 'red',
    'DINO': 'darkgreen',
    'MOCO': 'limegreen',
    'MAE': 'blue',
    'BEIT': 'darkturquoise',
    'RANDOM': 'magenta',
}
# POOLED_METHODS = {
#     'TIMM': 'firebrick',
#     'CLIP': 'lightcoral',
#     'DINO': 'olivedrab',
#     'MOCO': 'yellowgreen',
#     'MAE': 'steelblue',
#     'BEIT': 'lightskyblue',
# }
SIMPLE_MARKERS = {
    'BEIT': 'o',
    'CLIP': 'D',
    'DINO': '^',
    'MAE': 'h',
    'TIMM': 's',
    'MOCO': 'v',
    'RANDOM': '',
}
# note - runs using dense extractor mode append '-dense' to
# the end of the mod_id. This suffix is ignored
def get_line_fmt(mod_id, reduce_sat=0.75):
    if '-dense' in mod_id:
        mod_id = mod_id.replace('-dense','')
    if mod_id in POOLED_METHODS:
        c = POOLED_METHODS[mod_id]
        m = SIMPLE_MARKERS[mod_id]
        l = 'solid'
    else:
        m = ''
        id_parts = mod_id.split('-')
        if id_parts[0] not in SIMPLE_MARKERS:
            print('WARNING: no marker config for mod_id: ' + mod_id)
        else:
            m = SIMPLE_MARKERS[id_parts[0]]
        if mod_id not in SIMPLE_COLORS:
            print('WARNING: no plot config for mod_id: ' + mod_id)
            return 'k', m, 'solid'
        else:
            c = SIMPLE_COLORS[mod_id]
        # linestyle
        l = 'solid'
        if id_parts[3] in SIMPLE_LINESTYLES:
            l = SIMPLE_LINESTYLES[id_parts[3]]
    if reduce_sat < 1:
        c = matplotlib.colors.to_rgb(c)
        c = matplotlib.colors.rgb_to_hsv(c)
        c[1] *= reduce_sat
        c = matplotlib.colors.hsv_to_rgb(c)
    return c, m, l



##################################################


'''
provide a consistent order for model ids, handle new ids
pass in a list of mod_ids that you intend to plot, and it will return
a new list in the correct consistent order. Unknown ids will be added
at the end of the list.

Update: re-ordering these to be grouped by supervision type
Fully Supervised: TIMM, CLIP
Self Supervised, Global Objective: MOCO, DINO
Self Supervised, Local Objective: MAE

note - runs using dense extractor mode append '-dense' to
the end of the mod_id. These will follow the same order
as if the '-dense' suffix was ignored 
'''
MOD_ID_ORDER = [
    'TIMM-ViT-S-32-224', 'TIMM-ViT-S-16-224', 'TIMM-ViT-B-32-224',
    'TIMM-ViT-B-16-224', 'TIMM-ViT-B-8-224', 'TIMM-ViT-L-16-224',
    'CLIP-ViT-B-32-224', 'CLIP-ViT-B-16-224', 'CLIP-ViT-L-14-224',
    'DINO-ViT-S-16-224', 'DINO-ViT-S-8-224', 'DINO-ViT-B-16-224', 'DINO-ViT-B-8-224',
    'MOCO-ViT-S-16-224', 'MOCO-ViT-B-16-224', 
    'MAE-ViT-B-16-224', 'MAE-ViT-L-16-224', 'MAE-ViT-H-14-224',
    'BEIT-ViT-B-16-224', 'BEIT-ViT-L-16-224',
    'RANDOM-ViT-B-16-224'
]

def get_mod_id_order(mod_ids):
    ret = []
    for m in MOD_ID_ORDER:
        if m in mod_ids:
            ret.append(m)
        ms = m + '-dense'
        if ms in mod_ids:
            ret.append(ms)
    for m in mod_ids:
        if m not in ret:
            print('WARNING: no plot ordering for mod_id: ' + m)
            ret.append(m)
    return ret



##################################################



# Modified version of random_colors from DINO repo
# Added a gap at the end of the spectrum to make the
# two ends more distinct. Removed shuffle
def sequential_colors(N, bright=True):
    """
    Generate sequential colors.
    """
    brightness = 1.0 if bright else 0.85
    hsv = [(i / (N+2), 1, brightness) for i in range(N+2)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = colors[:-2]
    return colors