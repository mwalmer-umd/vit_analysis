"""
###########################################################################
Plotting methods for various types of attention metrics

Written by: Matthew Walmer
###########################################################################
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from meta_utils.plot_format import sequential_colors, get_line_fmt, get_mod_id_order, display_names
from meta_utils.plotters import pooled_blockwise_comparison_plot, filter_models, breakout_blockwise_comparison_plot, pertype_blockwise_comparison_plot

Image.MAX_IMAGE_PIXELS = 933120000

# plots for head-level metrics. Use sub_sel to plot a subset of the blocks. Pass a list of ints
def head_level_plots(output_dir, mod_id, all_img_res, a_m, fs=20, sub_sel=None, sbars=True):
    if len(all_img_res.shape)==2:
        all_img_res = np.expand_dims(all_img_res, 0)
    if all_img_res.shape[0] == 1:
        sbars = False
    m = np.mean(all_img_res, axis=0)
    s = np.std(all_img_res, axis=0)
    x = np.arange(0, all_img_res.shape[2])
    colors = sequential_colors(m.shape[0], bright=False)
    markers = ["o", "^", "s", "D"]
    # first plot - all heads sorted
    fig = plt.figure(figsize=[9,6])
    ax = plt.axes()
    leg_words = [] # Legend
    leg_marks = []
    # background stdev bars first (if enabled):
    if sbars:
        for b in range(m.shape[0]):
            if sub_sel is not None and b not in sub_sel: continue
            srt_idx = np.argsort(m[b,:])
            y_m = m[b,srt_idx]
            y_s = s[b,srt_idx]        
            y_m_ps = y_m + (2 * y_s)
            y_m_ms = y_m - (2 * y_s)
            cr, cg, cb = colors[b]
            plt.fill_between(x, y_m_ms, y_m_ps, linewidth=0.0, color=(cr, cg, cb, 0.15))
    # main lines:
    for b in range(m.shape[0]):
        if sub_sel is not None and b not in sub_sel: continue
        srt_idx = np.argsort(m[b,:])
        y_m = m[b,srt_idx]
        mark = markers[b%len(markers)]
        l, = plt.plot(x, y_m, marker=mark, markersize=5, color=colors[b])
        leg_words.append('blk %i'%b)
        leg_marks.append(l)
    # Legend and Axis Markers
    leg = ax.legend(leg_marks, leg_words, loc='center right', ncol=1, frameon=False, fontsize=fs, bbox_to_anchor=(1.3, 0.5))
    plt.gcf().subplots_adjust(left=0.12, top=0.9, bottom=0.1, right=0.80)
    plt.xlabel('Attention Head (Sorted)')
    plt.ylabel(a_m)
    plt.title(mod_id)
    # Save
    fname = os.path.join(output_dir, "%s_%s.png"%(mod_id, a_m))
    if not sbars:
        fname = os.path.join(output_dir, "%s_%s-[NO-BARS].png"%(mod_id, a_m))
    plt.savefig(fname)
    plt.close()
    # Secondary Plots - average of heads and max of heads
    pooling_methods = ['AVG', 'MAX']
    for pm in pooling_methods:
        if pm == 'AVG':
            mp = np.mean(m, axis=1)
        else:
            mp = np.max(m, axis=1)
        x = np.arange(0, mp.shape[0])
        fig = plt.figure(figsize=[9,6])
        plt.plot(x, mp, marker='.', markersize=20)
        plt.xlabel('Block')
        plt.ylabel(a_m + ' [%s-OVER-HEADS]'%pm)
        plt.title(mod_id)
        fname = os.path.join(output_dir, "%s_%s-[%s-OVER-HEADS].png"%(mod_id, a_m, pm))
        plt.savefig(fname)
        plt.close()



# plots for block-level metrics
def block_level_plots(output_dir, mod_id, all_img_res, a_m, fs=20, sbars=False, suff=None):
    if len(all_img_res.shape)==1:
        all_img_res = np.expand_dims(all_img_res, 0)
    if all_img_res.shape[0] == 1:
        sbars = False
    m = np.mean(all_img_res, axis=0)
    s = np.std(all_img_res, axis=0)
    x = np.arange(0, m.shape[0])
    fig = plt.figure(figsize=[9,6])
    plt.plot(x, m, marker='.', markersize=20)
    if sbars:      
        m_ps = m + (2 * s)
        m_ms = m - (2 * s)
        plt.fill_between(x, m_ms, m_ps, linewidth=0.0, color=(0.0, 0.0, 1.0, 0.15))
    plt.xlabel('Block')
    plt.ylabel(a_m)
    plt.title(mod_id)
    fname = os.path.join(output_dir, "%s_%s"%(mod_id, a_m))
    if suff is not None:
        fname += '-[%s]'%suff
    fname += ".png"
    plt.savefig(fname)
    plt.close()



def vs_plot(output_dir, mod_id, results, am1, am2, fs=20, fn_override=None):
    res1 = results[am1]
    if len(res1.shape) > 2:
        res1 = np.mean(res1, axis=0)
    res2 = results[am2]
    if len(res2.shape) > 2:
        res2 = np.mean(res2, axis=0)
    colors = sequential_colors(res1.shape[0], bright=False)
    markers = ["o", "^", "s", "D"]
    fig = plt.figure(figsize=[9,6])
    ax = plt.axes()
    leg_words = [] # Legend
    leg_marks = []
    for b in range(res1.shape[0]):
        rb1 = res1[b]
        rb2 = res2[b]
        mark = markers[b%len(markers)]
        l = plt.scatter(rb1, rb2, marker=mark, s=20, color=colors[b])
        leg_words.append('blk %i'%b)
        leg_marks.append(l)
    # Legend + Labels
    leg = ax.legend(leg_marks, leg_words, loc='center right', ncol=1, frameon=False, fontsize=fs, bbox_to_anchor=(1.3, 0.5))
    plt.gcf().subplots_adjust(left=0.12, top=0.95, bottom=0.1, right=0.80)
    plt.xlabel(am1)
    plt.ylabel(am2)
    plt.title(mod_id)
    # Save
    if fn_override is not None:
        fname = os.path.join(output_dir, "%s_%s.png"%(mod_id, fn_override))
    else:
        fname = os.path.join(output_dir, "%s_VS-plot_%s-[VS]-%s.png"%(mod_id, am1, am2))
    plt.savefig(fname)
    plt.close()
    


# simple nearest neighbor upscaling of numpy arrays. PIL had issues
# only for 2d arrays
def np_nn_upscale(a, sf=10):
    s0 = a.shape[0] * sf
    s1 = a.shape[1] * sf
    r = np.zeros((s0,s1), dtype=a.dtype)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            r[i*sf:(i+1)*sf,j*sf:(j+1)*sf] = a[i,j]
    return r



# is sort_order is true, will re-order each group of heads to be sorted by the metric, or
# if pre_order is provided, will order in the order set by it.
def summary_plot(output_dir, mod_id, results, a_m, sf=10, sort_order=False, pre_order=None):
    # consistent heatmap bounds per-metric
    heatmap_bounds = {
        'avg-att-dist': [0.0, 0.65],
        # 'stdev-over-token-pos': [0.0, 0.095],
        'spatial-cls-att': [0.0, 0.07],
        'avg-att-offset': [0.0, 0.086]
    }
    if len(results.shape)==2:
        results = np.expand_dims(results, 0)
    nb = results.shape[1]
    nh = results.shape[2]
    res = np.mean(results, axis=0)
    srt_ord = None
    if sort_order:
        if pre_order is not None:
            srt_ord = pre_order
        else:
            srt_ord = np.zeros_like(res, dtype=int)
            for i in range(nb):
                srt_idx = np.argsort(res[i,:])
                srt_ord[i,:] = srt_idx
        res_srt = np.zeros_like(res)
        for i in range(nb):
            srt_idx = srt_ord[i,:]
            for j in range(nh):
                res_srt[i,j] = res[i,srt_idx[j]]
        res = res_srt
    res = np_nn_upscale(res, sf)
    # per-metric min-max controls
    vmin = None
    vmax = None
    if a_m in heatmap_bounds:
        vmin = heatmap_bounds[a_m][0]
        vmax = heatmap_bounds[a_m][1]
    if sort_order:
        fname = os.path.join(output_dir, "%s_summary_%s.png"%(mod_id, a_m))
    else:
        fname = os.path.join(output_dir, "%s_summary_%s-[UNSORTED].png"%(mod_id, a_m))


    plt.imsave(fname=fname, arr=res, format='png', cmap='plasma', vmax=vmax, vmin=vmin, origin='upper')
    return srt_ord



# a visualization plot for metrics that computer a per-token metric
# if pre_scale == True, will normalize the scaling of all attention maps before plotting. this is necessary
# if there is a super-intense attention map which makes the scaling too extreme to see details in the others.
# the flag pre_shaped should be used if the metric is already reshaped to a square token grid
# cls_size controls how large the cls_token is rendered if include_cls = True
def token_plot(output_dir, mod_id, results, a_m, sf=10, buff=2, include_cls=False, pre_scale=False, pre_shaped=False, cls_size=2):
    nb = results.shape[0]
    nh = results.shape[1]
    nt = results.shape[2]
    if not pre_shaped:
        edge_len = int(np.sqrt(results.shape[2]-1))
        if edge_len * edge_len != (results.shape[2] - 1):
            print('ERROR: token_plot requires square token layout')
            exit(-1)
    else:
        edge_len = results.shape[2]
    met_rows = []
    mask_rows = []
    hbuff = None
    vbuff = None
    mbuff = None
    for b in range(nb):
        met_row = []
        mask_row = []
        for h in range(nh):
            # handle shaping of spatial tokens
            if not pre_shaped:
                spc_vis = np.reshape(results[b,h,1:], [edge_len, edge_len])
            else:
                spc_vis = results[b,h,:,:]
            # (optional) embed cls token in top-right of the spatial grid
            if include_cls:
                cls_vis = np.ones([edge_len+cls_size, edge_len+cls_size], dtype=results.dtype) * results[b,h,0]
                cls_vis[cls_size:,cls_size:] = spc_vis
                spc_vis = cls_vis
            # (optional) normalize map scalings first
            if pre_scale:
                spc_min = np.min(spc_vis)
                spc_max = np.max(spc_vis)
                div = (spc_max - spc_min)
                if div > 0:
                    spc_vis = (spc_vis - spc_min) / (spc_max - spc_min)
            # handle buffers and masks
            if hbuff is None:
                if include_cls:
                    hbuff = np.zeros([edge_len+cls_size, buff], dtype=results.dtype)
                else:
                    hbuff = np.zeros([edge_len, buff], dtype=results.dtype)
            if mbuff is None:
                if include_cls:
                    mbuff = np.ones([edge_len+cls_size, edge_len+cls_size], dtype=results.dtype)
                    mbuff[cls_size:,:cls_size] = 0
                    mbuff[:cls_size,cls_size:] = 0
                else:
                    mbuff = np.ones([edge_len, edge_len], dtype=results.dtype)
            # add to row
            met_row += [spc_vis, hbuff]
            mask_row += [mbuff, hbuff]
        # remove extra buffers
        met_row.pop(-1)
        mask_row.pop(-1)
        # finish row
        met_row = np.concatenate(met_row, axis=1)
        mask_row = np.concatenate(mask_row, axis=1)
        # vertical buffer
        if vbuff is None:
            vbuff = np.zeros([buff, met_row.shape[1]], dtype=results.dtype)
        # add to rows
        met_rows += [met_row, vbuff]
        mask_rows += [mask_row, vbuff]
    # remove extra buffer
    met_rows.pop(-1)
    mask_rows.pop(-1)
    # finish images
    met = np.concatenate(met_rows, axis=0)
    mask = np.concatenate(mask_rows, axis=0)
    # apply scaling factor
    met = np_nn_upscale(met, sf)
    mask = np_nn_upscale(mask, sf)
    # save image
    out_name = os.path.join(output_dir, "%s_tokenplot_%s.png"%(mod_id, a_m))
    plt.imsave(fname=out_name, arr=met, format='png')
    # load image, apply mask, save again
    im = np.array(Image.open(out_name))
    mask = mask.astype(im.dtype)
    imask = (1-mask)*255
    new_im = np.zeros_like(im)
    for i in range(3):
        new_im[:,:,i] = (mask * im[:,:,i]) + imask
    new_im[:,:,3] = 255
    new_im = Image.fromarray(new_im)
    new_im.save(out_name)



# utility to help load all data results for meta plots. dst_type specifies the desired
# pooling format. Currently supported: 'block' and 'network'
# added new mode head_p for head metrics that have already been pooled on the image_dimension
# when using "best-head" mode, will max-pool the head dimension instead of mean-pooling
def meta_dataload(cache_dir, mod_ids, a_m, a_m_type, dst_type, best_head=False):
    assert a_m_type in ['head', 'block', 'head_p', 'block_p']
    assert dst_type in ['head', 'block', 'network']
    need_expand = False
    if a_m_type == 'head_p':
        need_expand = True
        a_m_type = 'head'
    if a_m_type == 'block_p':
        need_expand = True
        a_m_type = 'block'
    if best_head and a_m_type != 'head':
        print('WARNING: meta_dataload best_head mode can only be used with a_m_type = block or block_p')
    if best_head and dst_type != 'block':
        print('WARNING: meta_dataload best_head mode can only be used with dst_type = block')
    id_ord = get_mod_id_order(mod_ids)
    all_data = []
    for mod_id in id_ord:
        # NEW universal cache format:
        cache_file = os.path.join(cache_dir, mod_id, '%s.npy'%(a_m))
        # old format for backwards compatibility:
        if not os.path.isfile(cache_file):
            cache_file = os.path.join(cache_dir, mod_id, '%s_%s_cache.npy'%(mod_id, a_m))
        data = np.load(cache_file)
        if need_expand:
            data = np.expand_dims(data, axis=0)
        # compress data dimensions
        if dst_type == 'head': 
            if a_m_type != 'head':
                print('ERROR: meta_dataload cannot convert from %s to %s'%(a_m_type, dst_type))
                exit(-1)
            data = np.mean(data, axis=0) # average along image dimension
        else:
            if a_m_type == 'head':
                if best_head:
                    data = np.max(data, axis=2) # choose the "best head"
                else:
                    data = np.mean(data, axis=2) # average along head dimension
            data = np.mean(data, axis=0) # average along image dimension
            if dst_type == 'network':
                data = np.mean(data, axis=0) # average along block dimension
        all_data.append(data)
    return id_ord, all_data



# load cached results for all models and combine into a single plot but average together the models
# from each group. These models can have different depths, so in those cases use interpolation.
# if breakout=True, creates breakout plots instead
# if best_head=True, uses max pooling instead of mean pooling over the head dimension
# if pertype=True, creates a pertype-style plot instead. Overrides breakout
def meta_plot(output_dir, cache_dir, mod_ids, a_m, a_m_type, inc_filters=None, exc_filters=None, breakout=False, best_head=False, pertype=False, **kwargs):
    # filter models before loadings
    mod_ids = filter_models(mod_ids, inc_filters=inc_filters, exc_filters=exc_filters)
    # load and plot results
    if pertype:
        id_ord, all_data = meta_dataload(cache_dir, mod_ids, a_m, a_m_type, 'block', best_head=best_head)
        pertype_blockwise_comparison_plot(output_dir, all_data, id_ord, a_m, inc_filters=inc_filters, exc_filters=exc_filters, **kwargs)
    elif breakout:
        id_ord, all_data = meta_dataload(cache_dir, mod_ids, a_m, a_m_type, 'head')
        breakout_blockwise_comparison_plot(output_dir, all_data, id_ord, a_m, inc_filters=inc_filters, exc_filters=exc_filters, **kwargs)
    else:
        id_ord, all_data = meta_dataload(cache_dir, mod_ids, a_m, a_m_type, 'block', best_head=best_head)
        pooled_blockwise_comparison_plot(output_dir, all_data, id_ord, a_m, inc_filters=inc_filters, exc_filters=exc_filters, **kwargs)

        

# load a metric, reduce it down to a single, network-level average value, and produce a vs plot for it
# and another metric
def meta_vs_plot(output_dir, cache_dir, mod_ids, a_m1, a_m1_type, a_m2, a_m2_type, fs=8):
    id_ord, a_m1_data = meta_dataload(cache_dir, mod_ids, a_m1, a_m1_type, 'network')
    _, a_m2_data = meta_dataload(cache_dir, mod_ids, a_m2, a_m2_type, 'network')
    # make plot
    fig = plt.figure(figsize=[9,6])
    ax = plt.axes()
    plt.scatter(a_m1_data, a_m2_data)
    for i, m_id in enumerate(id_ord):
        plt.annotate(m_id, (a_m1_data[i], a_m2_data[i]))
    # axis and title
    plt.xlabel(a_m1)
    plt.ylabel(a_m2)
    plt.title('%s VS %s (Network Averages)'%(a_m1, a_m2))
    # Save
    fname = os.path.join(output_dir, "META-VS-plot_%s-[VS]-%s.png"%(a_m1, a_m2))
    plt.savefig(fname)
    plt.close()