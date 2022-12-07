# Tools for making plots
import os
import math

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import interp1d

from meta_utils.plot_format import get_line_fmt, get_mod_id_order, display_names, metric_display_names, axis_marker_info, font_scale_info, is_bottom_plot


'''
Data management helper to filter model ids and results using include/exclude
filter strings
'''
def filter_models(mod_ids, all_data=None, inc_filters=None, exc_filters=None, silent=False):
    # inputs may be strings of lists of strings
    do_filtering = False
    if inc_filters is not None:
        do_filtering = True
        if not isinstance(inc_filters, list):
            inc_filters = [inc_filters]
    else:
        inc_filters = []
    if exc_filters is not None:
        do_filtering = True
        if not isinstance(exc_filters, list):
            exc_filters = [exc_filters]
    else:
        exc_filters = []
    # do filtering
    if not do_filtering:
        if all_data is None:
            return mod_ids
        else:
            return mod_ids, all_data
    else:
        keep_data = []
        keep_mods = []
        for i in range(len(mod_ids)):
            mod_id = mod_ids[i]
            keep = True
            for inc_f in inc_filters:
                if inc_f not in mod_id:
                    keep = False
            for exc_f in exc_filters:
                if exc_f in mod_id:
                    keep = False
            if keep:
                if all_data is not None:
                    keep_data.append(all_data[i])
                keep_mods.append(mod_ids[i])
        if not silent:
            if len(keep_mods) == 0:
                print('WARNING: None of the mod_ids provided satisfied the specified filters')
            else:
                print('Found %i mod_ids that satisfied filters'%len(keep_mods))
        if all_data is None:
            return keep_mods
        else:
            return keep_mods, keep_data



# from: https://stackoverflow.com/questions/4534480/get-legend-as-a-separate-picture-in-matplotlib
def export_legend(legend, filename="legend.png", grow=0.02):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if grow > 0.0:
        x0, y0, x1, y1 = bbox.extents
        x0 -= grow
        y0 -= grow
        x1 += grow
        y1 += grow
        bbox = matplotlib.transforms.Bbox.from_bounds(x0, y0, x1-x0, y1-y0)
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)



'''
A tool for comparing multiple models with a metric that is computed
separately for each block.

x-axis: block depth (normalized) or block number
y-axis: metric data (input)

If pooled == False, all lines will be plotted separated. If pooled == True
models belonging to the same group (DINO, MAE, etc) will be average-pooled
together into a single line. If the models have different depths, interpolation
will be used before averging them. Faded versions of the original lines will
be plotted behind the pooled lines

Inputs:
output_dir - where the plot will be saved
all_data - a list of 1D numpy arrays representing block-wise metric values for a given model
mod_ids - a list of model ids (strings) corresponding to the data
a_m - name of the analysis method/metric being plotted, for axis labels
pooled - enable/disable pooling by model type (default=True)
fs - (optional) base font size (different entries will be scaled)
inc_filters - (optional) include filters: either a string or list of strings. if specified, 
                will only plot results mod_ids that include (all) the specified string(s). Default = None
exc_filters - (optional) exclude filters: either a string or list of strings. if specified, 
                will NOT plot results mod_ids that include (any of) the specified string(s). Default = None
suff - (optional) add a custom suffix to the end of the file name
base_colors - if True, will force the line colors of individual models to batch the base model type color
                mainly for use with B-16-only plots
separate_legend - if True, will save the legend as a separate plot, in "thin" mode
x_block - use block numbers on x-axis instead of normalized block depth (for B-16 plots)
no_arch - if true, will hide the arch names (B/16, S/8, etc) in the legend
'''
def pooled_blockwise_comparison_plot(output_dir, all_data, mod_ids, a_m, pooled=True, fs=25, inc_filters=None,
        exc_filters=None, suff=None, base_colors=False, separate_legend=False, x_block=False, no_arch=False):
    plt.rcParams.update({
        # "text.usetex": True,
        "font.family": "sans-serif",
    })

    # result filtering
    mod_ids, all_data = filter_models(mod_ids, all_data, inc_filters, exc_filters, silent=True)
    # sort model id ordering
    all_data_dict = {}
    for i in range(len(mod_ids)):
        all_data_dict[mod_ids[i]] = all_data[i]
    id_ord = get_mod_id_order(mod_ids)
    # pooling by model type
    if pooled:
        methods = []
        pooled_data = {}
        for i, mod_id in enumerate(id_ord):
            method = mod_id.split('-')[0]
            if method not in pooled_data:
                methods.append(method)
                pooled_data[method] = []
            pooled_data[method].append(all_data_dict[mod_id])
        all_pooled_data = []
        for m in methods:
            data = pooled_data[m]
            if len(data) == 1:
                all_pooled_data.append(data[0])
            else:
                # ViT versions have different depths, so use interpolation before averaging:
                data_i = []
                for d in data:
                    d_i = interp_data(d)
                    data_i.append(d_i)
                data_i = np.stack(data_i, axis=0)
                data_i = np.mean(data_i, axis=0)
                all_pooled_data.append(data_i)
        pooled_id_ord = methods

    # make plot
    fig = plt.figure(figsize=[7.5,7])
    ax = plt.axes()
    leg_words = [] # Legend
    leg_marks = []
    leg_elements = []
    max_blocks = 0

    # label scaling:
    lab_sf = font_scale_info(a_m)
    mark_fs = fs*lab_sf

    # figure positioning:
    pos_sf = (lab_sf - 1)*0.6 + 1
    left = 0.18 * pos_sf
    top = 1.0 - (0.1 * pos_sf)
    bottom = 0.15 * pos_sf
    right = 0.99

    # title and axis super-labels
    plt_title, a_m_yname = metric_display_names(a_m)
    suptitle_x = ((right-left)/2)+left
    # special case
    if plt_title == 'Spat. Attention Saliency':
        suptitle_x -= 0.008
    fig.suptitle(plt_title, fontsize=mark_fs, x=suptitle_x)
    fig.supylabel(a_m_yname, fontsize=mark_fs, y=((top-bottom)/2)+bottom)
    if not x_block:
        fig.supxlabel('Layer Depth (Normalized)', fontsize=mark_fs, x=((right-left)/2)+left)
    else:
        fig.supxlabel('Layer', fontsize=mark_fs, x=((right-left)/2)+left)

    # unpooled lines:
    has_random = False
    alpha = 1.0
    if pooled:
        alpha = 0.1
    for m in range(len(all_data)):
        mod_id = id_ord[m]
        y = all_data_dict[id_ord[m]]
        # random: average together the first 10 entries as 10 random trials
        if 'RANDOM' in mod_id:
            has_random = True
            ym = np.mean(y[:10])
            y[:] = ym
        nb = all_data_dict[id_ord[m]].shape[0]
        if nb > max_blocks:
            max_blocks = nb
        if pooled or base_colors:
            col, mrk, lst = get_line_fmt(mod_id.split('-')[0])
        else:
            col, mrk, lst = get_line_fmt(mod_id)
        if not x_block:
            x = np.linspace(0, 1, num=nb)
        else:
            x = range(1, nb+1)
        l, = plt.plot(x, y, marker=mrk, markersize=7, color=col, linestyle=lst, alpha=alpha, linewidth=3)
        if not pooled:
            dn = display_names(mod_id, no_arch)
            leg_words.append(dn)
            leg_marks.append(l)
            if mrk == '':
                leg_e = Line2D([0], [0], marker=mrk, color=col, markersize=14, lw=3, label=dn)
            else:
                leg_e = Line2D([0], [0], marker=mrk, color=col, markersize=14, lw=3, label=dn, linestyle='None')
            leg_elements.append(leg_e)

    # pooled lines:
    if pooled:
        for m in range(len(all_pooled_data)):
            x = np.linspace(0, 1, num=all_pooled_data[m].shape[0])
            col, mrk, lst = get_line_fmt(pooled_id_ord[m])
            l, = plt.plot(x, all_pooled_data[m], marker=mrk, markersize=7, color=col, linestyle=lst, linewidth=3)
            dn = display_names(pooled_id_ord[m], no_arch)
            leg_words.append(dn)
            leg_marks.append(l)
            leg_e = Line2D([0], [0], marker=mrk, color=col, markersize=14, lw=3, label=dn, linestyle='None')
            leg_elements.append(leg_e)

    # Legend (on plot)
    plt.gcf().subplots_adjust(left=left, top=top, bottom=bottom, right=right)
    if not separate_legend:
        bottom = 0.1
        if pooled:
            leg = ax.legend(leg_elements, leg_words, loc='center right', ncol=1, frameon=False, fontsize=mark_fs, bbox_to_anchor=(1.12, 0.5))
        else:
            leg = ax.legend(leg_elements, leg_words, loc='center right', ncol=1, frameon=False, fontsize=mark_fs, bbox_to_anchor=(1.3, 0.5))    
    
    # Axis Markers
    if not x_block:
        # plt.xlabel('Layer Depth (Normalized)', fontsize=mark_fs)
        plt.xticks([0,1])
    else:
        # plt.xlabel('Layer', fontsize=mark_fs)
        if max_blocks == 12:
            x_marks, x_min, x_max = axis_marker_info('b16_blocks')
            plt.xticks(x_marks)
            plt.xlim(x_min, x_max)
        else:
            plt.xticks(range(1, max_blocks+1))
    y_axis_info = axis_marker_info(a_m)
    if y_axis_info is not None:
        y_marks, y_min, y_max = y_axis_info
        plt.yticks(y_marks)
        plt.ylim(y_min, y_max)
    plt.xticks(fontsize=mark_fs)
    plt.yticks(fontsize=mark_fs)
    plt_title, a_m_yname = metric_display_names(a_m)
    # plt.ylabel(a_m_yname, fontsize=mark_fs)
    # plt.title(plt_title, fontsize=mark_fs)
    plt.grid(True)

    # Save
    fname = os.path.join(output_dir, "blk-comparison-plot_%s"%a_m)
    if pooled:
        fname += '_[POOLED]'
    if suff is not None:
        fname += '_[%s]'%suff
    plt.savefig(fname + '.png')
    plt.savefig(fname + '.pdf')
    plt.savefig(fname + '.svg')

    # Legend (separate)
    # https://stackoverflow.com/questions/4534480/get-legend-as-a-separate-picture-in-matplotlib
    if separate_legend:
        leg_fs_mult = 1.0
        columnspacing = 1.0
        handletextpad = 0.3
        matplotlib.rcParams['legend.handlelength'] = 0.5
        # single row legend:
        leg = ax.legend(leg_elements, leg_words, loc='upper center', ncol=len(leg_words), frameon=False, framealpha=1.0, fontsize=fs*leg_fs_mult, bbox_to_anchor=(0.0, 3.5), columnspacing=columnspacing, handletextpad=handletextpad)
        plt.gcf().subplots_adjust(left=0.1, top=0.5, bottom=0.1, right=0.9)
        fig.supylabel('')
        # single column legend:
        # leg = ax.legend(leg_elements, leg_words, loc='center right', ncol=1, frameon=False, framealpha=1.0, fontsize=fs*leg_fs_mult, bbox_to_anchor=(3.5, 0.0), columnspacing=columnspacing, handletextpad=handletextpad)
        # plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.5)
        # fig.supxlabel('')
        if has_random:
            leg_fname = fname.replace("blk-comparison-plot_%s"%a_m,'legend_with_random')
        else:
            leg_fname = fname.replace("blk-comparison-plot_%s"%a_m,'legend')
        export_legend(leg, filename=leg_fname + '.png')
        export_legend(leg, filename=leg_fname + '.pdf')
        export_legend(leg, filename=leg_fname + '.svg')
    plt.close()





# helper to interpolate points for averaging models of different depths.
def interp_data(data, count=500):
    x = np.linspace(0, 1, num=data.shape[0])
    interper = interp1d(x, data, kind='linear')
    x_i = np.linspace(0, 1, num=count)
    data_i = np.zeros_like(x_i, dtype=float)
    data_i = interper(x_i)
    return data_i



BREAKOUT_PLOT_LIMITS = {
    'avg-att-dist' : ([0.1, 0.3, 0.5], 0.0, 0.67),
    'b16_blocks' : ([2, 8], 0.8, 12.2),
}

'''
A partner to pooled_blockwise_comparison_plot, but show each model in a separate breakout plot and separately
plot head head to give a better sense of the variation within blocks.

Note: the inputs "pooled" and "separated_legend" are kept only as placeholders and do not impact the plot

Legend will always be separate
'''
def breakout_blockwise_comparison_plot(output_dir, all_data, mod_ids, a_m, fs=25, inc_filters=None, exc_filters=None,
        suff=None, base_colors=False, x_block=False, no_arch=False, pooled=False, separate_legend=False):
    plt.rcParams.update({
        # "text.usetex": True,
        "font.family": "sans-serif",
    })

    # result filtering
    mod_ids, all_data = filter_models(mod_ids, all_data, inc_filters, exc_filters, silent=True)
    # sort model id ordering
    all_data_dict = {}
    for i in range(len(mod_ids)):
        all_data_dict[mod_ids[i]] = all_data[i]
    id_ord = get_mod_id_order(mod_ids)

    # label scaling:
    lab_sf = font_scale_info(a_m)
    mark_fs = fs*lab_sf

    # figure positioning:
    pos_sf = (lab_sf - 1)*0.6 + 1
    left = 0.18 * pos_sf
    top = 1.0 - (0.1 * pos_sf)
    bottom = 0.15 * pos_sf
    right = 0.99

    # prep figure
    if len(all_data) != 6:
        print('WARNING: breakout_blockwise_comparison_plot was designed for 6 sub-plots')
        return
    # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    # fig = plt.figure(figsize=[11,7])
    fig = plt.figure(figsize=[7.5,7])
    gs = fig.add_gridspec(2, 3, hspace=0, wspace=0)
    (ax1, ax2, ax3), (ax4, ax5, ax6) = gs.subplots(sharex='col', sharey='row')
    axs = [ax1, ax4, ax2, ax5, ax3, ax6]

    # title and axis super-labels
    plt_title, a_m_yname = metric_display_names(a_m)
    plt_title = plt_title.replace('(Layer)', '(Head)')
    fig.suptitle(plt_title, fontsize=mark_fs, x=((right-left)/2)+left)
    fig.supylabel(a_m_yname, fontsize=mark_fs, y=((top-bottom)/2)+bottom)
    if not x_block:
        fig.supxlabel('Layer Depth (Normalized)', fontsize=mark_fs, x=((right-left)/2)+left)
    else:
        fig.supxlabel('Layer', fontsize=mark_fs, x=((right-left)/2)+left)

    # plot data:
    for m in range(len(all_data)):
        ax = axs[m]

        # get data
        mod_id = id_ord[m]
        mod_id_base = mod_id.split('-')[0]
        y = all_data_dict[id_ord[m]]
        nb = y.shape[0]
        nh = y.shape[1]
        
        # get line settings
        if base_colors:
            col, mrk, lst = get_line_fmt(mod_id.split('-')[0])
        else:
            col, mrk, lst = get_line_fmt(mod_id)
        
        # get x markers
        if not x_block:
            x = np.linspace(0, 1, num=nb)
        else:
            x = range(1, nb+1)
        
        # plot heads
        for i in range(nh):
            ax.plot(x, y[:,i], marker=mrk, markersize=7, color=col, linestyle='', alpha=0.75, linewidth=3)
        
        # plot mean line
        ym = np.mean(y, axis=1)
        ax.plot(x, ym, marker='', color=col, linestyle=lst, alpha=1.0, linewidth=3)

        # axis arkers
        if not x_block:
            ax.set_xlabel('Layer Depth (Normalized)', fontsize=mark_fs)
            ax.set_xticks([0,1], fontsize=mark_fs)
            ax.set_xticklabels([0,1], fontsize=mark_fs)
        else:
            # ax.set_xlabel('Layer', fontsize=mark_fs)
            if nb == 12:
                x_marks, x_min, x_max = BREAKOUT_PLOT_LIMITS['b16_blocks']
                ax.set_xticks(x_marks, fontsize=mark_fs)
                ax.set_xticklabels(x_marks, fontsize=mark_fs)
                ax.set_xlim(x_min, x_max)
            else:
                ax.set_xticks(range(1, nb+1), fontsize=mark_fs)
                ax.set_xticklabels(range(1, nb+1), fontsize=mark_fs)
        if a_m in BREAKOUT_PLOT_LIMITS:
            y_marks, y_min, y_max = BREAKOUT_PLOT_LIMITS[a_m]
            ax.set_yticks(y_marks, fontsize=mark_fs)
            ax.set_yticklabels(y_marks, fontsize=mark_fs)
            ax.set_ylim(y_min, y_max)
        ax.grid(True)

    # Save
    plt.gcf().subplots_adjust(left=left, top=top, bottom=bottom, right=right)
    fname = os.path.join(output_dir, "breakout-comparison-plot_%s"%(a_m))
    if suff is not None:
        fname += '_[%s]'%suff
    plt.savefig(fname + '.png')
    plt.savefig(fname + '.pdf')
    plt.savefig(fname + '.svg')



'''
Another companion plotter to pooled_blockwise_comparison_plot, designed for plotting all results at once.
Will break the results for each supervision method into a separate plot.

Note: the inputs "pooled" and "separated_legend" are kept only as placeholders and do not impact the plot

Legend will always be integrated and placed under each separate sub-plot
'''
def pertype_blockwise_comparison_plot(output_dir, all_data, mod_ids, a_m, pooled=True, fs=25, inc_filters=None,
        exc_filters=None, suff=None, base_colors=False, separate_legend=False, x_block=False, no_arch=False):
    plt.rcParams.update({
        # "text.usetex": True,
        "font.family": "sans-serif",
    })
    methods = ['TIMM', 'CLIP', 'DINO', 'MOCO', 'MAE', 'BEIT']

    # warnings
    if x_block:
        print('WARNING: pertype_blockwise_comparison_plot is intended to be used with x_block=False')
    if base_colors:
        print('WARNING: pertype_blockwise_comparison_plot is intended to be used with base_colors=False')
    if no_arch:
        print('WARNING: pertype_blockwise_comparison_plot is intended to be used with no_arch=False')

    # result filtering
    mod_ids, all_data = filter_models(mod_ids, all_data, inc_filters, exc_filters, silent=True)
    # sort model id ordering
    all_data_dict = {}
    for i in range(len(mod_ids)):
        all_data_dict[mod_ids[i]] = all_data[i]
    id_ord = get_mod_id_order(mod_ids)

    # is bottom plot? include legend and xaxis labels only for bottom plots
    is_bottom = is_bottom_plot(a_m)
    
    # label scaling:
    # lab_sf = font_scale_info(a_m)
    lab_sf = 2.03 # override scaling factor
    if(a_m == 'Retrieval_roxford5k_mapH'):
        lab_sf = 1.8
    mark_fs = fs*lab_sf

    # y-axis mark info:
    y_settings = axis_marker_info(a_m, pertype=True)

    # figure positioning:
    left = 0.05
    top = 0.86
    bottom = 0.35
    right = 0.997
    title_y = 0.98
        
    # prep figure
    if not is_bottom:
        ibsf = 0.69916328
        fig = plt.figure(figsize=[45,13.5*ibsf])
        bottom = 0.07
        top = 1.0 - ((1.0 - top) * (1.0 / ibsf))
        title_y = 1.0 - ((1.0 - title_y) * (1.0 / ibsf))
    else:
        fig = plt.figure(figsize=[45,13.5])
    gs = fig.add_gridspec(1, 6, hspace=0, wspace=0.05)
    if y_settings is not None:
        ax1, ax2, ax3, ax4, ax5, ax6 = gs.subplots(sharey='row')
    else:
        ax1, ax2, ax3, ax4, ax5, ax6 = gs.subplots()
    axs = [ax1, ax2, ax3, ax4, ax5, ax6]

    # title and axis super-labels
    plt_title, a_m_yname = metric_display_names(a_m, pertype=True)
    fig.suptitle(plt_title, fontsize=mark_fs, x=((right-left)/2)+left, y=title_y)
    fig.supylabel(a_m_yname, fontsize=mark_fs, y=((top-bottom)/2)+bottom, x=0.001)
    if is_bottom:
        if not x_block:
            fig.supxlabel('Layer Depth (Normalized)', fontsize=mark_fs, x=((right-left)/2)+left, y=0.23)
        else:
            fig.supxlabel('Layer', fontsize=mark_fs, x=((right-left)/2)+left, y=0.23)

    # plot data:
    for m in range(len(methods)):
        method = methods[m]
        ax = axs[m]
        leg_words = []
        leg_elements = []

        for mod_id in id_ord:
            if 'RANDOM' not in mod_id and method not in mod_id: continue

            # get data
            y = all_data_dict[mod_id]

            # random: average together the first 10 entries as 10 random trials
            if 'RANDOM' in mod_id:
                ym = np.mean(y[:10])
                y[:] = ym
            
            # get line settings
            if base_colors:
                col, mrk, lst = get_line_fmt(mod_id.split('-')[0])
            else:
                col, mrk, lst = get_line_fmt(mod_id)
            
            # get x markers
            nb = y.shape[0]
            if not x_block:
                x = np.linspace(0, 1, num=nb)
            else:
                x = range(1, nb+1)
            
            # plot
            ax.plot(x, y, marker=mrk, markersize=7, color=col, linestyle=lst, alpha=1.0, linewidth=3)

            # legend entries
            dn = display_names(mod_id, no_sup=True, no_dense=True)
            leg_e = Line2D([0], [0], marker=mrk, color=col, linestyle=lst, markersize=14, lw=3, label=dn)
            if 'RANDOM' in mod_id:
                # only include random in right-most legend
                if m+1 == len(methods):
                    leg_words.append('Random')
                    leg_elements.append(leg_e)
            else:
                leg_words.append(dn)
                leg_elements.append(leg_e)
            
        # add legend for subplot
        if is_bottom:
            ncol = 1
            if len(leg_words) >= 4:
                ncol = 2
            leg_fs_mult = 1.0
            columnspacing = 1.0
            handletextpad = 0.3
            matplotlib.rcParams['legend.handlelength'] = 2.0
            leg = ax.legend(leg_elements, leg_words, loc='upper center', ncol=ncol, frameon=False, fontsize=mark_fs*leg_fs_mult, bbox_to_anchor=(0.5, -0.2), columnspacing=columnspacing, handletextpad=handletextpad)

        # axis markers
        ax.set_title(display_names(method), fontsize=mark_fs)
        if not x_block:
            # ax.set_xlabel('Layer Depth (Normalized)', fontsize=mark_fs)
            ax.set_xticks([0,0.5,1], fontsize=mark_fs)
            ax.set_xticklabels([0,0.5,1], fontsize=mark_fs)
        else:
            # ax.set_xlabel('Layer', fontsize=mark_fs)
            if nb == 12:
                x_marks, x_min, x_max = BREAKOUT_PLOT_LIMITS['b16_blocks']
                ax.set_xticks(x_marks, fontsize=mark_fs)
                ax.set_xticklabels(x_marks, fontsize=mark_fs)
                ax.set_xlim(x_min, x_max)
            else:
                ax.set_xticks(range(1, nb+1), fontsize=mark_fs)
                ax.set_xticklabels(range(1, nb+1), fontsize=mark_fs)
        if y_settings is not None:
            y_marks, y_min, y_max = y_settings
            ax.set_yticks(y_marks, fontsize=mark_fs)
            ax.set_yticklabels(y_marks, fontsize=mark_fs)
            ax.set_ylim(y_min, y_max)
        ax.grid(True)

    # Save
    plt.gcf().subplots_adjust(left=left, top=top, bottom=bottom, right=right)
    fname = os.path.join(output_dir, "pertype-comparison-plot_%s"%(a_m))
    if suff is not None:
        fname += '_[%s]'%suff
    plt.savefig(fname + '.png')
    plt.savefig(fname + '.pdf')
    plt.savefig(fname + '.svg')