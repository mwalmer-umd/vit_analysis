"""
###########################################################################
Tool to generate large grouped attention visualization plots

Written by: Matthew Walmer
###########################################################################
"""
import os
import argparse
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

Image.MAX_IMAGE_PIXELS = 933120000

NAME_MAPPER = {'TIMM' : 'FS', 'MOCO' : 'MoCo', 'BEIT' : 'BEiT'}

# refactor an input image grid to a new patch and buffer size
# using nearest neighbor sampling
# if extra_px == True, will add an extra row of pixels before resizing. This
# is usefull for the AAAM plots
def refactor_grid(im, im_patch, im_buff, new_patch, new_buff, extra_px=False):
    np_0 = int((int(im.shape[0]) + im_buff) / (im_patch + im_buff))
    np_1 = int((int(im.shape[1]) + im_buff) / (im_patch + im_buff))
    rows = []
    step = im_patch + im_buff
    h_buff = None
    v_buff = None
    for b in range(np_0):
        row = []
        for h in range(np_1):
            sl = im[b*step:b*step+im_patch, h*step:h*step+im_patch, :]
            if extra_px:
                temp = np.ones([sl.shape[0]+1, sl.shape[1]+1, sl.shape[2]], dtype=sl.dtype) * 255
                temp[:sl.shape[0], :sl.shape[1], :] = sl
                sl = temp
            sl = Image.fromarray(sl)
            sl = sl.resize([new_patch, new_patch], Image.Resampling.NEAREST)
            sl = np.array(sl)
            row.append(sl)
            if h_buff is None:
                h_buff = np.ones([sl.shape[0], new_buff, sl.shape[2]], dtype=sl.dtype) * 255
            row.append(h_buff)
        row.pop(-1)
        row = np.concatenate(row, axis=1)
        rows.append(row)
        if v_buff is None:
            v_buff = np.ones([new_buff, row.shape[1], row.shape[2]], dtype=row.dtype) * 255
        rows.append(v_buff)
    rows.pop(-1)
    rows = np.concatenate(rows, axis=0)
    return rows



def make_caption(text, fs, h, w, c, d):
    cap = np.ones([h, w, c], dtype=d) * 255
    cap = Image.fromarray(cap)
    idraw = ImageDraw.Draw(cap)
    font = ImageFont.truetype("analysis/FreeMono.ttf", fs)
    _, _, t_w, t_h = idraw.textbbox([0,0], text, font=font)
    idraw.text(((w-t_w)/2,(h-t_h)/2), text, fill="black", font=font)
    return np.array(cap)



# add a caption below an image
def add_caption(im, text, fs=150, h=150, on_top=False, on_left=False):
    if on_top:
        h = int(h*1.2)
    if on_left:
        w = h
        cap = np.ones([im.shape[0], w, im.shape[2]], dtype=im.dtype) * 255
    else:
        cap = np.ones([h, im.shape[1], im.shape[2]], dtype=im.dtype) * 255
    cap = Image.fromarray(cap)
    idraw = ImageDraw.Draw(cap)
    font = ImageFont.truetype("analysis/FreeMono.ttf", fs)
    # https://stackoverflow.com/questions/1970807/center-middle-align-text-with-pil
    _, _, t_w, t_h = idraw.textbbox([0,0], text, font=font)
    if on_left: # right justified
        idraw.text(((w-t_w), (im.shape[0]-t_h)/2), text, fill="black", font=font)
    else: # center on both dims
        idraw.text(((im.shape[1]-t_w)/2,(h-t_h)/2), text, fill="black", font=font)
    cap = np.array(cap)
    if on_left:
        ret = np.concatenate([cap, im], axis=1)
    elif on_top:
        ret = np.concatenate([cap, im], axis=0)
    else:
        ret = np.concatenate([im, cap], axis=0)
    return ret



# draw midlines on a token image
def add_midlines(im, t=4, soft=True):
    m = int(im.shape[0] / 2)
    tm = int(t/2)
    if not soft:
        im[:,m-tm:m+tm,1:2] = 0
        im[m-tm:m+tm,:,1:2] = 0
        im[:,m-tm:m+tm,0] = 255
        im[m-tm:m+tm,:,0] = 255
    else:
        data = im[:,m-tm:m+tm,0]
        temp = np.ones_like(data, dtype=float) * 255
        im[:,m-tm:m+tm,0] = ((temp + data) / 2).astype(im.dtype)
        data = im[m-tm:m+tm,:,0]
        temp = np.ones_like(data, dtype=float) * 255
        im[m-tm:m+tm,:,0] = ((temp + data) / 2).astype(im.dtype)
        # im[:,m-tm:m+tm,0] += 20
        # im[m-tm:m+tm,:,0] += 20
    return im



def make_sel_box(plots, sel, im_patch=140, im_buff=20, buff=20, scale_factor=2, fs_mult=1,
        cap_base_scale=20, bh_base_scale=15, make_captions=True):
    step = im_patch + im_buff
    slices = []
    h_buff = None
    for s in sel['sel']:
        m, b, h = s
        # special mode: load the input image and include it (prepended before the others)
        # the special setting should be placed last on the list of selections, but will
        # be inserted at the start of the slices instead
        if os.path.isfile(m):
            im_dim = slices[0].shape[1]
            im = Image.open(m).resize([im_dim, im_dim], Image.Resampling.NEAREST)
            im = np.array(im)
            if make_captions:
                im = add_caption(im, 'INPUT', int(bh_base_scale*scale_factor*fs_mult), int(bh_base_scale*scale_factor*fs_mult))
                im = add_caption(im, '', int(bh_base_scale*scale_factor*fs_mult), int(bh_base_scale*scale_factor*fs_mult))
            slices.insert(0, h_buff)
            slices.insert(0, im)
            continue
        im = plots[m]
        sl = im[b*step:b*step+im_patch, h*step:h*step+im_patch, :]
        if scale_factor != 1:
            sl = Image.fromarray(sl)
            sl = sl.resize([int(sl.size[0]*scale_factor), int(sl.size[1]*scale_factor)], Image.Resampling.NEAREST)
            sl = np.array(sl)
        if sel['midlines']:
            sl = add_midlines(sl)
        if make_captions:
            capm = m.split('-')[0]
            if capm in NAME_MAPPER:
                capm = NAME_MAPPER[capm]
            # cap = '%s B:%i H:%i'%(capm, b, h)
            # cap = '%s[%i,%i]'%(capm, b, h)
            # sl = add_caption(sl, cap, int(bh_base_scale*scale_factor*fs_mult), int(bh_base_scale*scale_factor*fs_mult))
            sl = add_caption(sl, capm, int(bh_base_scale*scale_factor*fs_mult), int(bh_base_scale*scale_factor*fs_mult))
            cap = '[%i,%i]'%(b+1, h+1) # Note - using 1-index for block and head names in final plots
            sl = add_caption(sl, cap, int(bh_base_scale*scale_factor*fs_mult), int(bh_base_scale*scale_factor*fs_mult))

        slices.append(sl)
        if h_buff is None:
            h_buff_w = int(buff*scale_factor)
            if 'pixel_steal' in sel and sel['pixel_steal'] > 0:
                h_buff_w -= sel['pixel_steal']
            h_buff = np.ones([sl.shape[0], h_buff_w, sl.shape[2]], dtype=sl.dtype) * 255
        slices.append(h_buff)
    slices.pop(-1)
    sel_box = np.concatenate(slices, axis=1)
    if make_captions:
        sel_box = add_caption(sel_box, sel['name'], int(cap_base_scale*scale_factor*fs_mult), int(cap_base_scale*scale_factor*fs_mult), on_top=True)
    return sel_box



def make_sel_box_row(plots, sels, w, im_patch=140, im_buff=20, scale_factor=2, fs_mult=1):
    # add selection boxes
    sbs = []
    for sel in sels:
        sel_box = make_sel_box(plots, sel, im_patch, im_buff, im_buff, scale_factor, fs_mult)
        sbs.append(sel_box)
    ws = w
    for sb in sbs:
        ws -= sb.shape[1]
    # gather and buffer selection boxes
    if len(sbs) == 1: # centered box with buffers on each side
        sb_buff = int(ws / (len(sbs) + 1))
        sb_buff_m = np.ones([sbs[0].shape[0], sb_buff, sbs[0].shape[2]], dtype=sbs[0].dtype) * 255
        ws -= (sb_buff * len(sbs))
        sb_buff_f = np.ones([sbs[0].shape[0], ws, sbs[0].shape[2]], dtype=sbs[0].dtype) * 255
        sb_row = []
        for sb in sbs:
            sb_row.append(sb_buff_m)
            sb_row.append(sb)
        sb_row.append(sb_buff_f)
    else: # multiple selections, justified left or right
        sb_buff = int(ws / (len(sbs) - 1))
        sb_buff_m = np.ones([sbs[0].shape[0], sb_buff, sbs[0].shape[2]], dtype=sbs[0].dtype) * 255
        ws -= (sb_buff * (len(sbs)-2))
        sb_buff_f = np.ones([sbs[0].shape[0], ws, sbs[0].shape[2]], dtype=sbs[0].dtype) * 255
        sb_row = []
        for sb in sbs:
            sb_row.append(sb)
            sb_row.append(sb_buff_m)
        sb_row.pop(-1)
        if len(sbs) > 2:
            sb_row[-2] = sb_buff_f
    sb_row = np.concatenate(sb_row, axis=1)
    return sb_row



# buff array a to be size "size" on axis "axis"
def buff_array(a, size, axis, center=False):
    assert axis in [0,1]
    s = np.array(a.shape)
    s[axis] = size
    ret = np.ones(s, a.dtype) * 255
    if center:
        sp = int((size - a.shape[axis])/2)
        if axis == 0:
            ret[sp:sp+a.shape[0], :a.shape[1], :] = a
        else:
            ret[:a.shape[0], sp:sp+a.shape[1], :] = a
    else:
        ret[:a.shape[0],:a.shape[1],:] = a
    return ret



# concatenate images along the specified axis, buffering the non-specified axis
def buff_concat(data, axis=0, center=False):
    assert axis in [0,1]
    if axis == 0:
        other = 1
    else:
        other = 0
    max_d = -1
    for d in data:
        dm = d.shape[other]
        if dm > max_d:
            max_d = dm
    buff_data = []
    for d in data:
        buff_data.append(buff_array(d, max_d, other, center))
    ret = np.concatenate(buff_data, axis=axis)
    return ret



'''
models = list of models to pull results for
a_m = analysis method for plots
sels = selected items to highlight in sub-boxes

if multiple patch scales are detected, will rescale all grids to a fixed size of 112 x 112

if im_patch = None, will guess the size based on the model parameters

sb_only - extra mode to save only the last selection-box row
'''
def grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch=None, im_buff=20, buff=40, attvis=False, fs_mult=1, refactor=True,
        downsize=1, sel_row_limit=1, sb_scale_factor=2, alg_plot=False, refactor_scaler=2, nh_inc=-1, title=None, row_lim=-1, prefix=None, sb_only=False, buffv=None, show_archp=False):
    os.makedirs(out_dir, exist_ok=True)
    if prefix is not None:
        out_name = os.path.join(out_dir, '%s_grouped_att_%s.png'%(prefix, a_m))
    else:
        out_name = os.path.join(out_dir, 'grouped_att_%s.png'%(a_m))
    print('analysis method: ' + a_m)
    # determine if multiple patch grid sizes are present
    compute_patch = (im_patch is None)
    pgss = []
    multiple_pgss = False
    for m in models:
        if '<PAD>' in m: continue
        p = int(m.split('-')[3])
        i = int(m.split('-')[4])
        pgs = int(i/p)
        if pgs not in pgss:
            pgss.append(pgs)
        if len(pgss) > 1:
            multiple_pgss = True
            break
    if show_archp:
        multiple_pgss = True
    # handle refactoring sizing
    if refactor:
        extra_px = False
        if multiple_pgss:
                new_im_patch = 112
                new_im_buff = 16
        else:
            if not alg_plot:
                new_im_patch = 28*refactor_scaler
                new_im_buff = 4*refactor_scaler
            else:
                new_im_patch = 27*refactor_scaler
                new_im_buff = 4*refactor_scaler
    # load grid images
    print('loading plots:')
    plots = {}
    for m in models:
        if '<PAD>' in m: continue
        if attvis:
            im_path = os.path.join(res_dir, m, 'head-v-pos', '%s_%s.png'%(m, a_m))
        else:
            im_path = os.path.join(res_dir, m, '%s_%s.png'%(m, a_m))
        im = np.array(Image.open(im_path))
        if compute_patch:
            p = int(m.split('-')[3])
            i = int(m.split('-')[4])
            pgs = int(i/p)
            if alg_plot:
                im_patch = ((pgs*2)-1)*10
            else:
                im_patch = pgs*10
        if refactor:
            im = refactor_grid(im, im_patch, im_buff, new_im_patch, new_im_buff, extra_px)
        plots[m] = im
        print('  ' + m + ' - ' + str(im.shape))
    # handle refactoring sizing
    if refactor:
        im_patch = new_im_patch
        im_buff = new_im_buff
    # generate output
    out_rows = []
    row = []
    h_buff = None
    v_buff = None
    if isinstance(row_lim, list):
        cur_row_lim = row_lim[0]
        row_lim.pop(0)
    else:
        cur_row_lim = row_lim
    for mc, m in enumerate(models):
        # pad
        if '<PAD>' in m:
            pad_w = int(int(m.split('-')[-1])/2)
            # note - cannot call a pad first
            pad = np.ones([row[0].shape[0], pad_w, row[0].shape[2]], dtype=row[0].dtype) * 255
            row.append(pad)
            row.append(pad)
        # actual content
        else:
            im = plots[m]
            if nh_inc != -1:
                nhi = (im_patch * nh_inc) + (im_buff * (nh_inc-1))
                im = im[:,:nhi,:]
            if downsize != 1:
                im = Image.fromarray(im)
                im = im.resize([int(im.size[0]/downsize), int(im.size[1]/downsize)], Image.Resampling.NEAREST)
                im = np.array(im)
            if multiple_pgss:
                cap = '%s/%s'%(m.split('-')[2], m.split('-')[3])
                im = add_caption(im, cap, 50*fs_mult, 50*fs_mult, on_top=True)
            cap = m.split('-')[0]
            if cap in NAME_MAPPER:
                cap = NAME_MAPPER[cap]
            im = add_caption(im, cap, 50*fs_mult, 50*fs_mult, on_top=True)
            row.append(im)
            if h_buff is None:
                h_buff = np.ones([im.shape[0], buff, im.shape[2]], dtype=im.dtype) * 255
            row.append(h_buff)
        # row completion
        if (len(row) == cur_row_lim * 2) or (mc+1 == len(models)):
            row.pop(-1)
            # row = np.concatenate(row, axis=1)
            row = buff_concat(row, axis=1)
            out_rows.append(row)
            row = []
            if v_buff is None:
                if buffv is None:
                    v_buff = np.ones([int(buff/2), out_rows[0].shape[1], out_rows[0].shape[2]], dtype=out_rows[0].dtype) * 255
                else:
                    v_buff = np.ones([buffv, out_rows[0].shape[1], out_rows[0].shape[2]], dtype=out_rows[0].dtype) * 255
            out_rows.append(v_buff)
            if isinstance(row_lim, list) and len(row_lim) > 0:
                cur_row_lim = row_lim[0]
                row_lim.pop(0)

    # early saving (no selection boxes)
    if len(sels) == 0:
        out_rows.pop(-1)
        if len(out_rows) == 1:
            out = out_rows[0]
        else:
            out = buff_concat(out_rows, axis=0, center=True)
        # title
        if title is not None:
            out = add_caption(out, title, 60*fs_mult, 60*fs_mult, on_top=True)
        print('saving: ' + out_name)
        Image.fromarray(out).save(out_name)
        return
    # sb only mode - just saving a selection box for additional visualizations
    if sb_only:
        sb = make_sel_box(plots, sels[0], im_patch, im_buff, im_buff, sb_scale_factor, fs_mult, make_captions=False)
        print('SB ONLY MODE')
        print('saving: ' + out_name)
        Image.fromarray(sb).save(out_name)
        return
    # sel box rows
    cur_sels = []
    for sel in sels:
        cur_sels.append(sel)
        if len(cur_sels) == sel_row_limit:
            sb_row = make_sel_box_row(plots, cur_sels, out_rows[0].shape[1], im_patch, im_buff, sb_scale_factor, fs_mult)
            out_rows.append(sb_row)
            out_rows.append(v_buff)
            cur_sels = []
    if len(cur_sels) > 0:
        sb_row = make_sel_box_row(plots, cur_sels, out_rows[0].shape[1], im_patch, im_buff, sb_scale_factor, fs_mult)
        out_rows.append(sb_row)
        out_rows.append(v_buff)
    out_rows.pop(-1)
    # out = np.concatenate(out_rows, axis=0)
    out = buff_concat(out_rows, axis=0, center=True)
    # title
    if title is not None:
        out = add_caption(out, title, 60*fs_mult, 60*fs_mult, on_top=True)
    # save
    print('saving: ' + out_name)
    Image.fromarray(out).save(out_name)
    # clear
    out = None
    cur_sels = None
    out_rows = None
    out = None
    plots = None



'''
models = list of models to pull results for
a_m = analysis method for plots
sels = selected items to highlight in sub-boxes
'''
def grouped_gif_plots(models, a_m, res_dir, out_dir, im_patch=140, im_buff=20, buff=8, attvis=False, fs_mult=1, refactor=True,
        downsize=1, sb_scale_factor=2, alg_plot=False, refactor_scaler=2, nh_inc=-1, title=None, tok_loc=None, blocks=None, prefix=None,
        teaser_mode=False):
    nh_inc = 1 # forced to one head
    os.makedirs(out_dir, exist_ok=True)
    print('analysis method: ' + a_m)
    # load grid images
    print('loading plots:')
    plots = {}
    for m in models:
        if attvis:
            im_path = os.path.join(res_dir, m, 'head-v-pos', '%s_%s.png'%(m, a_m))
        else:
            im_path = os.path.join(res_dir, m, '%s_%s.png'%(m, a_m))
        im = np.array(Image.open(im_path))
        if refactor:
            if not alg_plot:
                im = refactor_grid(im, im_patch, im_buff, 28*refactor_scaler, 4*refactor_scaler)
            else:
                im = refactor_grid(im, im_patch, im_buff, 27*refactor_scaler, 4*refactor_scaler)
        plots[m] = im
        print('  ' + m + ' - ' + str(im.shape))
    if refactor:
        if not alg_plot:
            im_patch = 28*refactor_scaler
            im_buff = 4*refactor_scaler
        else:
            im_patch = 27*refactor_scaler
            im_buff = 4*refactor_scaler
    # generate output
    nb = 12
    h_buff = None
    v_buff = None
    out_rows = []
    for m in models:
        row = []
        for b in range(nb):
            if blocks is not None and b not in blocks: continue
            im = plots[m]
            # select block/layer
            b_str = (im_patch + im_buff)*b
            im = im[b_str:b_str+im_patch,:,:]
            # sub-sample heads
            if nh_inc != -1:
                nhi = (im_patch * nh_inc) + (im_buff * (nh_inc-1))
                im = im[:,:nhi,:]
            # down-size
            if downsize != 1:
                im = Image.fromarray(im)
                im = im.resize([int(im.size[0]/downsize), int(im.size[1]/downsize)], Image.Resampling.NEAREST)
                im = np.array(im)
            row.append(im)
            # horizontal buffer
            if h_buff is None:
                h_buff = np.ones([row[0].shape[0], buff, row[0].shape[2]], dtype=row[0].dtype) * 255
            row.append(h_buff)
        row.pop(-1)
        if not teaser_mode:
            row.insert(0,h_buff)
        row = np.concatenate(row, axis=1)
        # add caption
        if not teaser_mode:
            cap = m.split('-')[0]
            if cap in NAME_MAPPER:
                cap = NAME_MAPPER[cap]
            cap_w = 60
            if tok_loc is not None:
                cap_w = 70
            row = add_caption(row, cap, 20*fs_mult, cap_w, on_left=True)
        # add row
        out_rows.append(row)
        row = []
        # vertical buffer
        if v_buff is None:
            v_buff = np.ones([buff, out_rows[0].shape[1], out_rows[0].shape[2]], dtype=out_rows[0].dtype) * 255
        out_rows.append(v_buff)
    out_rows.pop(-1)
    out = np.concatenate(out_rows, axis=0)
    # block header
    if blocks is None:
        block_header = []
        ex = out.shape[1] - (12 * (im_patch + im_buff)) + im_buff
        if ex > 0:
            block_header.append(make_caption('', 20*fs_mult, 20*fs_mult, ex, out.shape[2], out.dtype))
        block_header.append(make_caption('1', 20*fs_mult, 20*fs_mult, im_patch, out.shape[2], out.dtype))
        ex2 = out.shape[1] - ex - (2 * im_patch)
        block_header.append(make_caption('<- block ->', 20*fs_mult, 20*fs_mult, ex2, out.shape[2], out.dtype))
        block_header.append(make_caption('12', 20*fs_mult, 20*fs_mult, im_patch, out.shape[2], out.dtype))
        block_header = np.concatenate(block_header, axis=1)
    else:
        bh_buff = None
        block_header = []
        ex = out.shape[1] - (len(blocks) * (im_patch + im_buff)) + im_buff
        if ex > 0:
            block_header.append(make_caption('', 20*fs_mult, 20*fs_mult, ex, out.shape[2], out.dtype))
        for b in blocks:
            bc = make_caption(str(b+1), 20*fs_mult, 20*fs_mult, im_patch, out.shape[2], out.dtype)
            if bh_buff is None:
                bh_buff = np.ones([bc.shape[0], im_buff, out.shape[2]], out.dtype) * 255
            block_header.append(bc)
            block_header.append(bh_buff)
        block_header.pop(-1)
        block_header = np.concatenate(block_header, axis=1)
    if not teaser_mode:
        out = np.concatenate([block_header, v_buff, out], axis=0)
    else:
        out = np.concatenate([out, v_buff, block_header], axis=0)
    # token location display
    if not teaser_mode and tok_loc is not None:
        out = add_token_display(out, tok_loc, 14, 8, fs_mult)
    # title
    # if title is not None:
    #     out = add_caption(out, title, 30*fs_mult, 30*fs_mult, on_top=True)
    # save
    if prefix is not None:
        out_name = os.path.join(out_dir, '%s_%s.png'%(prefix,a_m))
    else:
        out_name = os.path.join(out_dir, 'grouped_gif_%s.png'%a_m)
    print('saving: ' + out_name)
    Image.fromarray(out).save(out_name)
    out_rows = []
    return



def add_token_display(im, tok_loc, tok_edge, sf, fs_mult, v_off=112):
    # token display
    td = np.ones([tok_edge, tok_edge, im.shape[2]], dtype=im.dtype) * 128
    tok_loc -= 1
    tr = int(math.floor(tok_loc / tok_edge))
    tc = tok_loc - (tr * tok_edge)
    td[tr,tc,0] = 255
    td[tr,tc,1] = 0
    td[tr,tc,2] = 0
    # upscale
    if sf > 0:
        td = Image.fromarray(td)
        d0 = int(td.size[0]*sf)
        d1 = int(td.size[1]*sf)
        td = td.resize([d0, d1], Image.Resampling.NEAREST)
        td = np.array(td)
    # caption
    td = add_caption(td, 'position', fs=20*fs_mult, h=20*fs_mult, on_top=True)
    td = add_caption(td, 'token', fs=20*fs_mult, h=20*fs_mult, on_top=True)
    # pad
    pad = np.ones([im.shape[0], td.shape[1], im.shape[2]], dtype=im.dtype) * 255
    s = v_off
    pad[s:s+td.shape[0],:,:] = td    
    # add
    ret = np.concatenate([pad, im], axis=1)
    return ret



# https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
def export_gif(out_dir, a_ms):
    images = []
    for a_m in a_ms:
        im_path = os.path.join(out_dir, 'grouped_gif_%s.png'%a_m)
        images.append(imageio.imread(im_path))
    out_name = os.path.join(out_dir, 'grouped_gif.gif')
    imageio.mimsave(out_name, images)



def pad_like(d1, d2):
    ret = np.ones_like(d2) * 255
    ret[:d1.shape[0],:,:] = d1
    return ret



def meta_merge(f1, f2, fdir, buff=30):
    d1 = np.array(Image.open(os.path.join(fdir,f1)))
    d2 = np.array(Image.open(os.path.join(fdir,f2)))
    if d2.shape[0] < d1.shape[0]:
        d2 = pad_like(d2, d1)
    elif d2.shape[0] > d1.shape[0]:
        d1 = pad_like(d1, d2)
    buff = np.ones([d1.shape[0], buff, d1.shape[2]], dtype=d1.dtype) * 255
    ret = np.concatenate([d1, buff, d2], axis=1)
    if ret.shape[2] == 4:
        ret = ret[:,:,:3]
    return ret





'''
manually pull the items specified in sels without resizing them.

Note, will only read the first sel in sels
'''
def manual_sample(models, a_m, sels, res_dir, out_dir, im_patch=None, im_buff=20, attvis=False, alg_plot=False):
    compute_patch = False
    if im_patch is None:
        compute_patch = True
    print('analysis method: ' + a_m)
    # load grid images
    print('loading plots:')
    plots = {}
    all_im_patch = {}
    all_im_buff = {}
    for m in models:
        if '<PAD>' in m: continue
        # load image
        if attvis:
            im_path = os.path.join(res_dir, m, 'head-v-pos', '%s_%s.png'%(m, a_m))
        else:
            im_path = os.path.join(res_dir, m, '%s_%s.png'%(m, a_m))
        im = np.array(Image.open(im_path))
        # derive patch grid size
        p = int(m.split('-')[3])
        i = int(m.split('-')[4])
        pgs = int(i/p)
        if compute_patch:
            if not alg_plot:
                im_patch = pgs*10
            else:
                im_patch = ((pgs*2)-1)*10
        all_im_patch[m] = im_patch
        all_im_buff[m] = im_buff
        plots[m] = im
        print('  ' + m + ' - ' + str(im.shape))
    # generate outputs
    out_dir_full = os.path.join(out_dir, 'attention_samples_%s.png'%(a_m))
    os.makedirs(out_dir_full, exist_ok=True)
    sel = sels[0]
    for s in sel['sel']:
        m, b, h = s
        im = plots[m]
        im_patch = all_im_patch[m]
        im_buff = all_im_buff[m]
        step = im_patch + im_buff
        sl = im[b*step:b*step+im_patch, h*step:h*step+im_patch, :]
        if sel['midlines']:
            sl = add_midlines(sl)
        fn = '%s_%i_%i.png'%(m, b, h)
        out_name = os.path.join(out_dir_full, fn)
        Image.fromarray(sl).save(out_name)



def insert_vpad(a, vp, w):
    l = a[:, :vp, :]
    r = a[:, vp:, :]
    p = np.ones([a.shape[0], w, a.shape[2]], dtype=a.dtype) * 255
    ret = np.concatenate([l,p,r], axis=1)
    return ret




# generate grouped_att_plots but separate the models by type. ignores sels. Forces row_lim to -1 (no limit), forces title to None, forces show_archp to True
def pertype_grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch=None, im_buff=20, buff=40, attvis=False, fs_mult=1, refactor=True,
        downsize=1, sel_row_limit=1, sb_scale_factor=2, alg_plot=False, refactor_scaler=2, nh_inc=-1, title=None, row_lim=-1, prefix=None, sb_only=False, buffv=None, show_archp=False):
    types = ['TIMM', 'CLIP', 'DINO', 'MOCO', 'MAE', 'BEIT']
    for t in types:
        new_models = []
        for m in models:
            if t in m:
                new_models.append(m)
        if prefix is None:
            new_prefix = 'pertype-%s'%t
        else:
            new_prefix = 'pertype-%s'%t + '-' + prefix
        grouped_att_plots(new_models, a_m, [], res_dir, out_dir, im_patch, im_buff, buff, attvis, fs_mult, refactor, downsize, sel_row_limit, sb_scale_factor, alg_plot,
            refactor_scaler, nh_inc, None, -1, new_prefix, sb_only, buffv, True)



################################################################################



def main(args):
    
    #################### B/16-ONLY PLOTS ####################
    models = [
        'TIMM-ViT-B-16-224',
        'CLIP-ViT-B-16-224',
        'DINO-ViT-B-16-224',
        'MOCO-ViT-B-16-224',
        'MAE-ViT-B-16-224',
        'BEIT-ViT-B-16-224',
    ]
    out_dir = 'attention_analysis_out/_att_highlights'
    nh_inc = 3 # number of heads to include. set to -1 to include all
    sel_row_limit = 1 # limit sel rows to this many
    sb_scale_factor = 3 # how much to scale up attention maps in the selection boxes
    
    ##### SINGLE IMAGE CLS ATTENTION #####
    
    if args.single_cls or args.all:
        title = '1 Image: CLS Attention'
        im_name = 'n02124075_9919_crop.jpg'
        im_patch= 375
        im_path = os.path.join('vis_in', im_name)
        a_m = '(img %s)_(blk YAXIS)_(pos cls)_(head XAXIS)'%im_name
        res_dir = 'vis_out'
        im_buff=10
        fs_mult=1
        sels = [
            {
                'name' : 'FS & CLIP: Sparse Repeating Patterns',
                'midlines' : False,
                'sel' : [
                    ('TIMM-ViT-B-16-224', 9, 0),
                    ('TIMM-ViT-B-16-224', 10, 1),
                    ('TIMM-ViT-B-16-224', 11, 2),
                    ('CLIP-ViT-B-16-224', 9, 0),
                    ('CLIP-ViT-B-16-224', 10, 1),
                    ('CLIP-ViT-B-16-224', 11, 2),
                ],
            },
            {
                'name' : 'DINO & MoCo: Object Centerness Blobs',
                'midlines' : False,
                'sel' : [
                    ('DINO-ViT-B-16-224', 9, 0),
                    ('DINO-ViT-B-16-224', 10, 0),
                    ('DINO-ViT-B-16-224', 11, 1),
                    ('MOCO-ViT-B-16-224', 9, 0),
                    ('MOCO-ViT-B-16-224', 10, 1),
                    ('MOCO-ViT-B-16-224', 11, 1),
                ],
            },
            {
                'name' : 'MAE & BEiT: Diverse Attention',
                'midlines' : False,
                'sel' : [
                    ('MAE-ViT-B-16-224', 9, 0),
                    ('MAE-ViT-B-16-224', 10, 1),
                    ('MAE-ViT-B-16-224', 11, 2),
                    ('BEIT-ViT-B-16-224', 9, 0),
                    ('BEIT-ViT-B-16-224', 10, 1),
                    ('BEIT-ViT-B-16-224', 11, 2),
                ],
            },
        ]
        grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, attvis=True, fs_mult=fs_mult, sel_row_limit=sel_row_limit, sb_scale_factor=sb_scale_factor, nh_inc=nh_inc, title=title)

    ##### SINGLE IMAGE SPC-AGG ATTENTION #####

    if args.single_spc or args.all:
        # Cat Image 1
        title = '1 Image: Avg Spatial Attention'
        im_name = 'n02124075_9919_crop.jpg'
        im_patch= 375
        im_path = os.path.join('vis_in', im_name)
        a_m = '(img %s)_(blk YAXIS)_(pos spcagg)_(head XAXIS)'%im_name
        res_dir = 'vis_out'
        im_buff=10
        fs_mult=1
        sels = [
            {
                'name' : 'FS & CLIP: Sparse Repeating Patterns',
                'midlines' : False,
                'sel' : [
                    ('TIMM-ViT-B-16-224', 9, 0),
                    ('TIMM-ViT-B-16-224', 10, 1),
                    ('TIMM-ViT-B-16-224', 11, 2),
                    ('CLIP-ViT-B-16-224', 9, 0),
                    ('CLIP-ViT-B-16-224', 10, 1),
                    ('CLIP-ViT-B-16-224', 11, 1),
                ],
            },
            {
                'name' : 'DINO & MoCo: Object Centerness Blobs',
                'midlines' : False,
                'sel' : [
                    ('DINO-ViT-B-16-224', 9, 0),
                    ('DINO-ViT-B-16-224', 10, 1),
                    ('DINO-ViT-B-16-224', 11, 2),
                    ('MOCO-ViT-B-16-224', 9, 2),
                    ('MOCO-ViT-B-16-224', 10, 1),
                    ('MOCO-ViT-B-16-224', 11, 1),
                ],
            },
            {
                'name' : 'MAE & BEiT: Diverse Attention',
                'midlines' : False,
                'sel' : [
                    ('MAE-ViT-B-16-224', 9, 0),
                    ('MAE-ViT-B-16-224', 10, 1),
                    ('MAE-ViT-B-16-224', 11, 2),
                    ('BEIT-ViT-B-16-224', 9, 0),
                    ('BEIT-ViT-B-16-224', 10, 1),
                    ('BEIT-ViT-B-16-224', 11, 2),
                ],
            },
            {
                'name' : 'Offset Local Attention',
                'midlines' : False,
                'sel' : [
                    ('TIMM-ViT-B-16-224', 2, 6),
                    ('CLIP-ViT-B-16-224', 3, 0),
                    ('DINO-ViT-B-16-224', 3, 0),
                    ('MOCO-ViT-B-16-224', 2, 1),
                    ('MAE-ViT-B-16-224', 4, 4),
                    ('BEIT-ViT-B-16-224', 1, 6),
                ],
            },
        ]
        grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, attvis=True, fs_mult=fs_mult, sel_row_limit=sel_row_limit, sb_scale_factor=sb_scale_factor, nh_inc=nh_inc, title=title)

    ##### MULTI IMAGE AGGREGATE CLS ATTENTION #####

    if args.agg_cls or args.all:
        title = '5k Images: CLS Attention'
        a_m = 'tokenplot_avg-cls-att-on-token-[PRE-SCALED]'
        res_dir = 'attention_analysis_out'
        im_patch=140
        im_buff=20
        fs_mult=1
        sels = [
            {
                'name' : 'FS & CLIP: Sparse Repeating Patterns',
                'midlines' : False,
                'sel' : [
                    ('TIMM-ViT-B-16-224', 9, 0),
                    ('TIMM-ViT-B-16-224', 10, 1),
                    ('TIMM-ViT-B-16-224', 11, 2),
                    ('CLIP-ViT-B-16-224', 9, 0),
                    ('CLIP-ViT-B-16-224', 10, 1),
                    ('CLIP-ViT-B-16-224', 11, 2),
                ],
            },
            {
                'name' : 'DINO & MoCo: Object Centerness Blobs',
                'midlines' : False,
                'sel' : [
                    ('DINO-ViT-B-16-224', 9, 0),
                    ('DINO-ViT-B-16-224', 10, 1),
                    ('DINO-ViT-B-16-224', 11, 2),
                    ('MOCO-ViT-B-16-224', 9, 0),
                    ('MOCO-ViT-B-16-224', 10, 1),
                    ('MOCO-ViT-B-16-224', 11, 2),
                ],
            },
            {
                'name' : 'MAE & BEiT: Diverse Attention',
                'midlines' : False,
                'sel' : [
                    ('MAE-ViT-B-16-224', 9, 0),
                    ('MAE-ViT-B-16-224', 10, 1),
                    ('MAE-ViT-B-16-224', 11, 2),
                    ('BEIT-ViT-B-16-224', 9, 0),
                    ('BEIT-ViT-B-16-224', 10, 1),
                    ('BEIT-ViT-B-16-224', 11, 2),
                ],
            },
        ]
        grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, fs_mult=fs_mult, sel_row_limit=sel_row_limit, sb_scale_factor=sb_scale_factor, nh_inc=nh_inc, title=title)

    ##### MULTI IMAGE AGGREGATE SPC-AGG ATTENTION #####

    if args.agg_spc or args.all:
        title = '5k Images: Avg Spatial Attention'
        # a_m = 'tokenplot_avg-att-on-token-[PRE-SCALED]'
        a_m = 'tokenplot_avg-spcpure-att-on-token-[PRE-SCALED]'
        res_dir = 'attention_analysis_out'
        im_patch=140
        im_buff=20
        fs_mult=1
        sels = [
            {
                'name' : 'FS & CLIP: Sparse Repeating Patterns',
                'midlines' : False,
                'sel' : [
                    ('TIMM-ViT-B-16-224', 9, 0),
                    ('TIMM-ViT-B-16-224', 10, 1),
                    ('TIMM-ViT-B-16-224', 11, 2),
                    ('CLIP-ViT-B-16-224', 9, 0),
                    ('CLIP-ViT-B-16-224', 10, 1),
                    ('CLIP-ViT-B-16-224', 11, 2),
                ],
            },
            {
                'name' : 'DINO & MoCo: Object Centerness Blobs',
                'midlines' : False,
                'sel' : [
                    ('DINO-ViT-B-16-224', 9, 0),
                    ('DINO-ViT-B-16-224', 10, 1),
                    ('DINO-ViT-B-16-224', 11, 2),
                    ('MOCO-ViT-B-16-224', 9, 0),
                    ('MOCO-ViT-B-16-224', 10, 1),
                    ('MOCO-ViT-B-16-224', 11, 2),
                ],
            },
            {
                'name' : 'MAE & BEiT: Diverse Attention',
                'midlines' : False,
                'sel' : [
                    ('MAE-ViT-B-16-224', 9, 0),
                    ('MAE-ViT-B-16-224', 10, 1),
                    ('MAE-ViT-B-16-224', 11, 2),
                    ('BEIT-ViT-B-16-224', 9, 0),
                    ('BEIT-ViT-B-16-224', 10, 1),
                    ('BEIT-ViT-B-16-224', 11, 2),
                ],
            },
            {
                'name' : 'Offset Local Attention',
                'midlines' : False,
                'sel' : [
                    ('TIMM-ViT-B-16-224', 2, 6),
                    ('CLIP-ViT-B-16-224', 3, 0),
                    ('DINO-ViT-B-16-224', 3, 0),
                    ('MOCO-ViT-B-16-224', 2, 1),
                    ('MAE-ViT-B-16-224', 4, 4),
                    ('BEIT-ViT-B-16-224', 1, 6),
                ],
            },
        ]
        grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, fs_mult=fs_mult, sel_row_limit=sel_row_limit, sb_scale_factor=sb_scale_factor, nh_inc=nh_inc, title=title)

    ##### MULTI IMAGE AGGREGATE ALIGNED SPC-AGG ATTENTION #####

    if args.aligned or args.all:
        # overriding some settings:
        # nh_inc = 6 # number of heads to include. set to -1 to include all
        # sel_row_limit = 2 # limit sel rows to this many
        # sb_scale_factor = 2 # how much to scale up attention maps in the selection boxes
        # nh_inc = 3 # number of heads to include. set to -1 to include all
        title = 'Average Aligned Spatial Attention'
        a_m = 'tokenplot_avg-aligned-att-on-token-[PRE-SCALED]'
        res_dir = 'attention_analysis_out'
        im_patch=270
        im_buff=20
        fs_mult=1
        sels = [
            {
                'name' : 'Strict Local Attention',
                'midlines' : False,
                'sel' : [
                    ('TIMM-ViT-B-16-224', 0, 6),
                    ('CLIP-ViT-B-16-224', 0, 0),
                    ('DINO-ViT-B-16-224', 0, 5),
                    ('MOCO-ViT-B-16-224', 1, 3),
                    ('MAE-ViT-B-16-224', 1, 8), 
                    ('BEIT-ViT-B-16-224', 1, 2),  
                ],
            },
            {
                'name' : 'Soft Local Attention',
                'midlines' : False,
                'sel' : [
                    ('TIMM-ViT-B-16-224', 2, 5),
                    ('CLIP-ViT-B-16-224', 1, 2),
                    ('DINO-ViT-B-16-224', 1, 8),
                    ('MOCO-ViT-B-16-224', 1, 5),
                    ('MAE-ViT-B-16-224', 0, 11),
                    ('BEIT-ViT-B-16-224', 2, 8), 
                ],
            },
            {
                'name' : 'Axial Local Attention',
                'midlines' : False,
                'sel' : [
                    ('TIMM-ViT-B-16-224', 4, 6),
                    ('CLIP-ViT-B-16-224', 4, 5),
                    ('DINO-ViT-B-16-224', 5, 2),
                    ('MOCO-ViT-B-16-224', 4, 4),
                    ('MAE-ViT-B-16-224', 5, 8),
                    ('BEIT-ViT-B-16-224', 2, 11),
                ],
            },
            {
                'name' : 'Offset Local Attention',
                'midlines' : True,
                'sel' : [
                    ('TIMM-ViT-B-16-224', 2, 6),
                    ('CLIP-ViT-B-16-224', 3, 0),
                    ('DINO-ViT-B-16-224', 3, 0),
                    ('MOCO-ViT-B-16-224', 2, 1),
                    ('MAE-ViT-B-16-224', 4, 4),
                    ('BEIT-ViT-B-16-224', 1, 6),
                ],
            },
        ]
        grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, fs_mult=fs_mult, sel_row_limit=sel_row_limit, sb_scale_factor=sb_scale_factor, alg_plot=True, nh_inc=nh_inc, title=title)

    ##### META COMBINED PLOT #####

    if args.meta or args.all:
        meta_buff_size = 163
        files = [
            'grouped_att_(img n02124075_9919_crop.jpg)_(blk YAXIS)_(pos cls)_(head XAXIS).png',
            'grouped_att_(img n02124075_9919_crop.jpg)_(blk YAXIS)_(pos spcagg)_(head XAXIS).png',
            'grouped_att_tokenplot_avg-cls-att-on-token-[PRE-SCALED].png',
            'grouped_att_tokenplot_avg-spcpure-att-on-token-[PRE-SCALED].png',
        ]
        r1 = meta_merge(files[0], files[1], out_dir, meta_buff_size)
        r2 = meta_merge(files[2], files[3], out_dir, meta_buff_size)
        rs = [r1, r2]
        for i in range(2):
            out_name = os.path.join(out_dir, 'meta_grouped_att_%i.png'%(i+1))
            print('saving: ' + out_name)
            Image.fromarray(rs[i]).save(out_name)
        # meta_buff = np.ones([int(meta_buff_size/2), r1.shape[1], r1.shape[2]], dtype=r1.dtype) * 255
        # meta_plot = np.concatenate([r1, meta_buff, r2])
        # out_name = os.path.join(out_dir, 'meta_grouped_att.png')
        # print('saving: ' + out_name)
        # Image.fromarray(meta_plot).save(out_name)

    ##### GIF VIEW / TEASER VIEW #####
    # special gif visualization
    if args.gif:
        im_name = 'n02124075_9919_crop.jpg'
        im_path = os.path.join('vis_in', im_name)
        # a_ms = [
        #     '(img %s)_(blk YAXIS)_(pos cls)_(head XAXIS)'%im_name,
        #     '(img %s)_(blk YAXIS)_(pos bottom)_(head XAXIS)'%im_name,
        #     '(img %s)_(blk YAXIS)_(pos center)_(head XAXIS)'%im_name,
        #     '(img %s)_(blk YAXIS)_(pos left)_(head XAXIS)'%im_name,
        #     '(img %s)_(blk YAXIS)_(pos right)_(head XAXIS)'%im_name,
        #     '(img %s)_(blk YAXIS)_(pos top)_(head XAXIS)'%im_name,
        # ]
        # titles = [
        #     'CLS Token',
        #     'Bottom Token',
        #     'Center Token',
        #     'Left Token',
        #     'Right Token',
        #     'Top Token',
        # ]
        
        # # for animated plot and teaser v1:
        # im_patch= 375
        # im_buff = 10
        # res_dir = 'vis_out'
        # a_ms = ['(img %s)_(blk YAXIS)_(pos cls)_(head XAXIS)'%im_name]
        # titles = ['CLS Token']
        # tok_loc = [None]
        # for i in range(14):
        #     a_ms.append('(img %s)_(blk YAXIS)_(pos diag%02i)_(head XAXIS)'%(im_name,i))
        #     titles.append('diagonal %02i'%i)
        #     tok_loc.append(i*15+1)
        
        # for teaser v2:
        im_patch=140
        im_buff=20
        res_dir = 'attention_analysis_out'
        a_ms = ['tokenplot_avg-cls-att-on-token-[PRE-SCALED]']
        titles = ['CLS Token']
        tok_loc = [None]
        
        # sub-select models
        # models = [
        #     'TIMM-ViT-B-16-224',
        #     'DINO-ViT-B-16-224',
        #     'MAE-ViT-B-16-224',
        # ]
        # sub-select blocks
        # blocks = None
        # blocks = [0, 5, 11]
        blocks = [0, 3, 7, 11]
        
        out_dir = 'attention_analysis_out/_att_highlights_gif'
        nh_inc = 1 # number of heads to include. set to -1 to include all
        fs_mult = 1
        for a_n in range(len(a_ms)):
            title = titles[a_n]
            a_m = a_ms[a_n]
            tl = tok_loc[a_n]
            # animated gif:
            # grouped_gif_plots(models, a_m, res_dir, out_dir, im_patch, im_buff, attvis=True, fs_mult=fs_mult, alg_plot=False, nh_inc=nh_inc, title=title, tok_loc=tl)
            # teaser v1.0
            # grouped_gif_plots(models, a_m, res_dir, out_dir, im_patch, im_buff, attvis=True, fs_mult=fs_mult, alg_plot=False, nh_inc=nh_inc, title=title, tok_loc=tl, blocks=blocks, prefix='teaser', teaser_mode=True)
            # teaser v2.0
            grouped_gif_plots(models, a_m, res_dir, out_dir, im_patch, im_buff, attvis=False, fs_mult=fs_mult, alg_plot=False, nh_inc=nh_inc, title=title, tok_loc=tl, prefix='teaser_v2', teaser_mode=True)

        # a_ms.pop(0)
        # export_gif(out_dir, a_ms)

    ##### ALL HEAD PLOTS (B/16) #####
    if args.all_heads: # non-aligned plots
        models = [
            'TIMM-ViT-B-16-224',
            'DINO-ViT-B-16-224',
            'MAE-ViT-B-16-224',
            'CLIP-ViT-B-16-224',
            'MOCO-ViT-B-16-224',
            'BEIT-ViT-B-16-224',
        ]
        nh_inc = -1 # number of heads to include. set to -1 to include all
        ##### SINGLE IMAGE CLS ATTENTION #####
        title = '1 Image: CLS Attention'
        im_name = 'n02124075_9919_crop.jpg'
        im_patch= 375
        im_path = os.path.join('vis_in', im_name)
        a_m = '(img %s)_(blk YAXIS)_(pos cls)_(head XAXIS)'%im_name
        res_dir = 'vis_out'
        im_buff=10
        fs_mult=1
        sels = []
        grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, attvis=True, fs_mult=fs_mult, sel_row_limit=sel_row_limit,
            sb_scale_factor=sb_scale_factor, nh_inc=nh_inc, title=title, row_lim=3, prefix='allheads')
        ##### SINGLE IMAGE SPC-AGG ATTENTION #####
        title = '1 Image: Avg Spatial Attention'
        im_name = 'n02124075_9919_crop.jpg'
        im_patch= 375
        im_path = os.path.join('vis_in', im_name)
        a_m = '(img %s)_(blk YAXIS)_(pos spcagg)_(head XAXIS)'%im_name
        res_dir = 'vis_out'
        im_buff=10
        fs_mult=1
        sels = []
        grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, attvis=True, fs_mult=fs_mult, sel_row_limit=sel_row_limit,
            sb_scale_factor=sb_scale_factor, nh_inc=nh_inc, title=title, row_lim=3, prefix='allheads')
        ##### MULTI IMAGE AGGREGATE CLS ATTENTION #####
        title = '5k Images: CLS Attention'
        a_m = 'tokenplot_avg-cls-att-on-token-[PRE-SCALED]'
        res_dir = 'attention_analysis_out'
        im_patch=140
        im_buff=20
        fs_mult=1
        sels = []
        grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, fs_mult=fs_mult, sel_row_limit=sel_row_limit,
            sb_scale_factor=sb_scale_factor, nh_inc=nh_inc, title=title, row_lim=3, prefix='allheads')
        ##### MULTI IMAGE AGGREGATE SPC-AGG ATTENTION #####
        title = '5k Images: Avg Spatial Attention'
        a_m = 'tokenplot_avg-spcpure-att-on-token-[PRE-SCALED]'
        res_dir = 'attention_analysis_out'
        im_patch=140
        im_buff=20
        fs_mult=1
        sels = []
        grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, fs_mult=fs_mult, sel_row_limit=sel_row_limit,
            sb_scale_factor=sb_scale_factor, nh_inc=nh_inc, title=title, row_lim=3, prefix='allheads')

    if args.all_heads_aligned:
        models = [
            'TIMM-ViT-B-16-224',
            'DINO-ViT-B-16-224',
            'MAE-ViT-B-16-224',
            'CLIP-ViT-B-16-224',
            'MOCO-ViT-B-16-224',
            'BEIT-ViT-B-16-224',
        ]
        nh_inc = -1 # number of heads to include. set to -1 to include all
        sel_row_limit = 2 # limit sel rows to this many
        sb_scale_factor = 3 # how much to scale up attention maps in the selection boxes
        title = 'Average Aligned Spatial Attention'
        a_m = 'tokenplot_avg-aligned-att-on-token-[PRE-SCALED]'
        res_dir = 'attention_analysis_out'
        im_patch=270
        im_buff=20
        fs_mult=1
        sels = []
        grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, fs_mult=fs_mult, sel_row_limit=sel_row_limit,
            sb_scale_factor=sb_scale_factor, alg_plot=True, nh_inc=nh_inc, title=title, row_lim=3, prefix='allheads')

    #################### SINGLE SAMPLE SELECTION PLOTS ####################
    

    if args.single_rows:
        # simple single row plots for single image inputs to be stacked together
        im_names = ['n02124075_9919_crop.jpg', 'ILSVRC2012_val_00013223.JPEG', 'ILSVRC2012_val_00014752_crop.jpg', 'ILSVRC2012_val_00019745_crop.jpg', 'ILSVRC2012_val_00024194_crop.jpg',
            'ILSVRC2012_val_00035690_crop.jpg', 'ILSVRC2012_val_00037723_crop.jpg', 'ILSVRC2012_val_00038819_crop.jpg', 'ILSVRC2012_val_00043615_crop.jpg', 'ILSVRC2012_val_00045189_crop.jpg']
        im_patches = [375, 500, 382, 480, 375, 375, 375, 375, 334, 413]
        im_list = []

        title = None
        res_dir = 'vis_out'
        out_dir = 'attention_analysis_out/_att_highlights_single_all_models'
        im_buff=10
        fs_mult=1
        models = [
            'TIMM-ViT-S-32-224',
            'TIMM-ViT-S-16-224',
            'TIMM-ViT-B-32-224',
            'TIMM-ViT-B-16-224',
            'TIMM-ViT-B-8-224',
            'TIMM-ViT-L-16-224',
            'CLIP-ViT-B-32-224',
            'CLIP-ViT-B-16-224',
            'CLIP-ViT-L-14-224',
            'DINO-ViT-S-16-224',
            'DINO-ViT-S-8-224' ,
            'DINO-ViT-B-16-224',
            'DINO-ViT-B-8-224' ,
            'MOCO-ViT-S-16-224',
            'MOCO-ViT-B-16-224',
            'MAE-ViT-B-16-224' ,
            'MAE-ViT-L-16-224' ,
            'MAE-ViT-H-14-224' ,
            'BEIT-ViT-B-16-224',
            'BEIT-ViT-L-16-224',
        ]
        sels = [
            {
                'name' : '',
                'midlines' : False,
                'sel' : [
                    ('TIMM-ViT-S-32-224', 11, 0),
                    ('TIMM-ViT-S-16-224', 11, 0),
                    ('TIMM-ViT-B-32-224', 11, 0),
                    ('TIMM-ViT-B-16-224', 11, 0),
                    ('TIMM-ViT-B-8-224' , 11, 0),
                    ('TIMM-ViT-L-16-224', 23, 0),
                    ('CLIP-ViT-B-32-224', 11, 0),
                    ('CLIP-ViT-B-16-224', 11, 0),
                    ('CLIP-ViT-L-14-224', 23, 0),
                    ('DINO-ViT-S-16-224', 11, 0),
                    ('DINO-ViT-S-8-224' , 11, 0),
                    ('DINO-ViT-B-16-224', 11, 0),
                    ('DINO-ViT-B-8-224' , 11, 0),
                    ('MOCO-ViT-S-16-224', 11, 0),
                    ('MOCO-ViT-B-16-224', 11, 0),
                    ('MAE-ViT-B-16-224' , 11, 0),
                    ('MAE-ViT-L-16-224' , 23, 0),
                    ('MAE-ViT-H-14-224' , 31, 0),
                    ('BEIT-ViT-B-16-224', 11, 0),
                    ('BEIT-ViT-L-16-224', 23, 0),
                ],
            },
            # {
            #     'name' : '',
            #     'midlines' : False,
            #     'sel' : [
            #         ('TIMM-ViT-S-32-224', 5, 0),
            #         ('TIMM-ViT-S-16-224', 5, 0),
            #         ('TIMM-ViT-B-32-224', 5, 0),
            #         ('TIMM-ViT-B-16-224', 5, 0),
            #         ('TIMM-ViT-B-8-224' , 5, 0),
            #         ('TIMM-ViT-L-16-224', 11, 0),
            #         ('CLIP-ViT-B-32-224', 5, 0),
            #         ('CLIP-ViT-B-16-224', 5, 0),
            #         ('CLIP-ViT-L-14-224', 11, 0),
            #         ('DINO-ViT-S-16-224', 5, 0),
            #         ('DINO-ViT-S-8-224' , 5, 0),
            #         ('DINO-ViT-B-16-224', 5, 0),
            #         ('DINO-ViT-B-8-224' , 5, 0),
            #         ('MOCO-ViT-S-16-224', 5, 0),
            #         ('MOCO-ViT-B-16-224', 5, 0),
            #         ('MAE-ViT-B-16-224' , 5, 0),
            #         ('MAE-ViT-L-16-224' , 11, 0),
            #         ('MAE-ViT-H-14-224' , 15, 0),
            #         ('BEIT-ViT-B-16-224', 5, 0),
            #         ('BEIT-ViT-L-16-224', 11, 0),
            #     ],
            # },
        ]
        for idx in range(len(im_names)):
            im_name = im_names[idx]
            im_patch = im_patches[idx]
            im_path = os.path.join('vis_in', im_name)
            a_m = '(img %s)_(blk YAXIS)_(pos cls)_(head XAXIS)'%im_name
            sels[0]['sel'].append((im_path, None, None))
            grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, attvis=True, fs_mult=fs_mult, sel_row_limit=sel_row_limit,
                sb_scale_factor=sb_scale_factor, nh_inc=nh_inc, title=title, sb_only=True)
            fn = os.path.join(out_dir, "grouped_att_%s.png"%a_m)
            im_list.append(fn)
            sels[0]['sel'].pop(-1)

        # 5k average as last row
        a_m = 'tokenplot_avg-cls-att-on-token-[PRE-SCALED]'
        res_dir = 'attention_analysis_out'
        im_patch=None
        im_buff=20
        grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, fs_mult=fs_mult, sel_row_limit=sel_row_limit,
            sb_scale_factor=sb_scale_factor, nh_inc=-1, title=title, prefix='allmodsandheads', row_lim=-1, buff=80, buffv=200, sb_only=True)
        fn = os.path.join(out_dir, "allmodsandheads_grouped_att_%s.png"%a_m)
        im_list.append(fn)

        # stack
        rows = []
        v_buff = None
        for im_fn in im_list:
            im = np.array(Image.open(im_fn))
            if 'avg-cls' in im_fn:
                temp1 = np.ones([im.shape[0], 336, im.shape[2]], dtype=im.dtype) * 128
                temp2 = np.ones([im.shape[0], 48, im.shape[2]], dtype=im.dtype) * 255
                im = np.concatenate([temp1, temp2, im], axis=1)
                im = im[:,:,:3]
            rows.append(im)
            if v_buff is None:
                v_buff = np.ones([48, im.shape[1], im.shape[2]], dtype=im.dtype) * 255
            rows.append(v_buff)
        rows.pop(-1)
        rows = np.concatenate(rows, axis=0)
        v_pads = [7279, 6122, 5350, 3816, 2667, 361]
        for vp in v_pads:
            rows = insert_vpad(rows, vp, 96)
        out_name = os.path.join(out_dir, 'rows_stacked.png')
        Image.fromarray(rows).save(out_name)



    #################### ALL MODEL SUPPLEMENTAL PLOTS ####################
    if args.all_mods:
        fs_mult = 4
        row_lim = -1
        sels = []
        models = [
            'TIMM-ViT-S-32-224',
            'TIMM-ViT-S-16-224',
            'TIMM-ViT-B-32-224',
            'TIMM-ViT-B-16-224',
            'TIMM-ViT-B-8-224',
            'TIMM-ViT-L-16-224',
            'CLIP-ViT-B-32-224',
            'CLIP-ViT-B-16-224',
            'CLIP-ViT-L-14-224',
            'DINO-ViT-S-16-224',
            'DINO-ViT-S-8-224' ,
            'DINO-ViT-B-16-224',
            'DINO-ViT-B-8-224' ,
            '<PAD>-200',
            'MOCO-ViT-S-16-224',
            'MOCO-ViT-B-16-224',
            'MAE-ViT-B-16-224' ,
            'MAE-ViT-L-16-224' ,
            'MAE-ViT-H-14-224' ,
            '<PAD>-200',
            'BEIT-ViT-B-16-224',
            'BEIT-ViT-L-16-224',
        ]
        ##### SINGLE IMAGE CLS ATTENTION #####
        title = '1 Image: CLS Attention'
        im_name = 'n02124075_9919_crop.jpg'
        im_path = os.path.join('vis_in', im_name)
        a_m = '(img %s)_(blk YAXIS)_(pos cls)_(head XAXIS)'%im_name
        res_dir = 'vis_out'
        im_patch= 375
        im_buff=10
        # grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, attvis=True, fs_mult=fs_mult, sel_row_limit=sel_row_limit,
        #     sb_scale_factor=sb_scale_factor, nh_inc=nh_inc, title=title, row_lim=row_lim, prefix='allmods', buff=80)
        # grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, fs_mult=fs_mult, sel_row_limit=sel_row_limit,
        #     sb_scale_factor=sb_scale_factor, nh_inc=-1, title=title, prefix='allmodsandheads', row_lim=7, buff=80)

        ##### SINGLE IMAGE SPC-AGG ATTENTION #####
        title = '1 Image: Avg Spatial Attention'
        im_name = 'n02124075_9919_crop.jpg'
        im_path = os.path.join('vis_in', im_name)
        a_m = '(img %s)_(blk YAXIS)_(pos spcagg)_(head XAXIS)'%im_name
        res_dir = 'vis_out'
        im_patch= 375
        im_buff=10
        # grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, attvis=True, fs_mult=fs_mult, sel_row_limit=sel_row_limit,
        #     sb_scale_factor=sb_scale_factor, nh_inc=nh_inc, title=title, row_lim=row_lim, prefix='allmods', buff=80)
        # grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, fs_mult=fs_mult, sel_row_limit=sel_row_limit,
        #     sb_scale_factor=sb_scale_factor, nh_inc=-1, title=title, prefix='allmodsandheads', row_lim=7, buff=80)
        
        ##### MULTI IMAGE AGGREGATE CLS ATTENTION #####
        title = '5k Images: CLS Attention'
        a_m = 'tokenplot_avg-cls-att-on-token-[PRE-SCALED]'
        res_dir = 'attention_analysis_out'
        im_patch=None
        im_buff=20
        sels = []
        # grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, fs_mult=fs_mult, sel_row_limit=sel_row_limit,
        #     sb_scale_factor=sb_scale_factor, nh_inc=nh_inc, title=title, row_lim=row_lim, prefix='allmods', buff=80)
        # with all heads:
        row_lim = [6, 3, 7, 6]
        pertype_grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, fs_mult=fs_mult, sel_row_limit=sel_row_limit,
            sb_scale_factor=sb_scale_factor, nh_inc=-1, title=title, prefix='allmodsandheads', row_lim=row_lim, buff=80, buffv=200)
        
        ##### MULTI IMAGE AGGREGATE SPC-AGG ATTENTION #####
        title = '5k Images: Avg Spatial Attention'
        a_m = 'tokenplot_avg-spcpure-att-on-token-[PRE-SCALED]'
        res_dir = 'attention_analysis_out'
        im_patch=None
        im_buff=20
        sels = []
        # grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, fs_mult=fs_mult, sel_row_limit=sel_row_limit,
        #     sb_scale_factor=sb_scale_factor, nh_inc=nh_inc, title=title, row_lim=row_lim, prefix='allmods', buff=80)
        # grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, fs_mult=fs_mult, sel_row_limit=sel_row_limit,
        #     sb_scale_factor=sb_scale_factor, nh_inc=-1, title=title, prefix='allmodsandheads', row_lim=7, buff=80)

        ##### AAAMs #####
        title = 'Average Aligned Spatial Attention'
        a_m = 'tokenplot_avg-aligned-att-on-token-[PRE-SCALED]'
        res_dir = 'attention_analysis_out'
        im_patch=None
        im_buff=20
        sels = []
        # grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, fs_mult=fs_mult, sel_row_limit=1,
        #     sb_scale_factor=sb_scale_factor, alg_plot=True, nh_inc=nh_inc, title=title, row_lim=row_lim, prefix='allmods', buff=60)
        # with all heads:
        row_lim = [6, 3, 7, 6]
        pertype_grouped_att_plots(models, a_m, sels, res_dir, out_dir, im_patch, im_buff, fs_mult=fs_mult, sel_row_limit=sel_row_limit,
            sb_scale_factor=sb_scale_factor, nh_inc=-1, title=title, prefix='allmodsandheads', row_lim=row_lim, buff=80, alg_plot=True, buffv=200)

        ##### Offset Local Attention Samples #####
        a_m = 'tokenplot_avg-aligned-att-on-token-[PRE-SCALED]'
        res_dir = 'attention_analysis_out'
        sels = [
            {
                'name' : '',
                'midlines' : True,
                'sel' : [
                    ('TIMM-ViT-S-32-224', 1, 1),
                    ('TIMM-ViT-S-16-224', 2, 5),
                    ('TIMM-ViT-B-32-224', 2, 2),
                    ('TIMM-ViT-B-16-224', 2, 6),
                    ('TIMM-ViT-B-8-224' , 2, 0),
                    ('TIMM-ViT-L-16-224', 1, 3),
                    ('CLIP-ViT-B-32-224', 2, 11),
                    ('CLIP-ViT-B-16-224', 2, 2),
                    ('CLIP-ViT-L-14-224', 6, 6),
                    ('DINO-ViT-S-16-224', 2, 1),
                    ('DINO-ViT-S-8-224' , 2, 2),
                    ('DINO-ViT-B-16-224', 2, 1),
                    ('DINO-ViT-B-8-224' , 2, 6),
                    ('MOCO-ViT-S-16-224', 3, 3),
                    ('MOCO-ViT-B-16-224', 2, 9),
                    ('MAE-ViT-B-16-224' , 10, 6),
                    ('MAE-ViT-L-16-224' , 6, 4), # diagonal
                    ('MAE-ViT-H-14-224' , 10, 2), # diagonal
                    ('BEIT-ViT-B-16-224', 1, 6),
                    ('BEIT-ViT-L-16-224', 21, 12),
                ],
            },
        ]
        manual_sample(models, a_m, sels, res_dir, out_dir, im_patch=None, im_buff=20, attvis=False, alg_plot=True)



################################################################################



if __name__ == '__main__':
    parser = argparse.ArgumentParser('group and select visualizations')
    # B/16 - only plots
    parser.add_argument('--single_cls', action='store_true')
    parser.add_argument('--single_spc', action='store_true')
    parser.add_argument('--agg_cls', action='store_true')
    parser.add_argument('--agg_spc', action='store_true')
    parser.add_argument('--aligned', action='store_true')
    parser.add_argument('--meta', action='store_true')
    parser.add_argument('--all', action='store_true')
    # GIF / teaser plots
    parser.add_argument('--gif', action='store_true')
    # Supplemental plots
    parser.add_argument('--all_heads', action='store_true')
    parser.add_argument('--all_heads_aligned', action='store_true')
    parser.add_argument('--all_mods', action='store_true')
    parser.add_argument('--single_rows', action='store_true')
    args = parser.parse_args()
    main(args)