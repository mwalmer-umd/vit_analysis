"""
###########################################################################
Gather attention visualizations into a meta grid with multiple options for
axis variables.

Written by: Matthew Walmer
###########################################################################
"""
import os
import argparse
import numpy as np
from PIL import Image



def parse_filename(f):
    f_par = f.split('+')
    if len(f_par) == 1: return None
    img, vtype, pos, head, blk = f_par
    blk = blk.split('.')[0] # remove extension
    f_info = {}
    f_info['img'] = img
    f_info['pos'] = pos
    f_info['head'] = head
    f_info['blk'] = blk
    f_info['vtype'] = vtype
    return f_info



def concat_with_pad(items, axis, pad):
    assert axis in [0,1]
    if axis == 1:
        pad_o = items[0].shape[0]
        blank = np.ones([pad_o, pad, 3], dtype=np.uint8) * 255
    else:
        pad_o = items[0].shape[1]
        blank = np.ones([pad, pad_o, 3], dtype=np.uint8) * 255
    new_items = []
    for i in items:
        new_items.append(i)
        new_items.append(blank)
    new_items.pop(-1)
    c = np.concatenate(new_items, axis=axis)
    return c



def load_image(fn, img_size=None):
    img = Image.open(fn)
    img = img.convert('RGB')
    if img_size is not None:
        img = img.resize([img_size, img_size])
    return np.array(img)



def special_order(axis_disc, var):
    if var == 'pos':
        return ['top', 'left', 'center', 'right', 'bottom']
    return axis_disc



########################################



'''
Gather images in an input dir into grid visualizations

INPUT/OUTPUT
    input_dir - Dir with exported attention map images
    output_dir - Dir to save grid images to
GRID VARIABLES
    xaxis - which variable to change on the xaxis of the grid
    yaxis - which variable to change on the yaxis of the grid
    use_all - (optional) apply every found option for this variable (different from xaxis and yaxis)
FIXED VARIABLES
    pos - fixed position, if not an axis variable or all variable
    blk - fixed block, if not an axis variable or all variable
    head - fixed head, if not an axis variable or all variable
    img - fixed image, if not an axis variable or all variable
OTHER
    img_size - Resize images to a square with this edge length (only if img is an axis variable)
    pad - padding between images
    scale_factor - scale factor for small version of image
    save_small - if enabled, save a downscaled copy also
'''
def make_supergrid(input_dir, output_dir, xaxis='pos', yaxis='blk', use_all=None, pos=None, blk=None, head=None,
        img=None, img_size=480, pad=10, scale_factor=0.125, mod_id=None, save_small=False):
    # check axis settings
    valid_variables = ['img', 'blk', 'pos', 'head']
    assert xaxis in valid_variables
    assert yaxis in valid_variables
    assert xaxis != yaxis
    if use_all is not None:
        assert use_all in valid_variables
        assert use_all != xaxis
        assert use_all != yaxis
    resize_needed = False
    if xaxis == 'img' or yaxis == 'img':
        resize_needed = True

    # gather images
    i = 0
    original_images = []
    yaxis_srt = {}
    xaxis_disc = []
    full_discovery = {}
    for v in valid_variables:
        full_discovery[v] = []
    for f in os.listdir(input_dir):
        f_info = parse_filename(f)
        if f_info is None:
            original_images.append(f)
            continue
        if f_info['vtype'] != 'attention':
            continue
        # discovery variable options
        for v in valid_variables:
            if f_info[v] not in full_discovery[v]:
                full_discovery[v].append(f_info[v])
        # sort by axis selection
        yaxis_info = f_info[yaxis]
        xaxis_info = f_info[xaxis]
        if yaxis_info not in yaxis_srt:
            yaxis_srt[yaxis_info] = {}
        if xaxis_info not in xaxis_disc:
            xaxis_disc.append(xaxis_info)
        if xaxis_info not in yaxis_srt[yaxis_info]:
            yaxis_srt[yaxis_info][xaxis_info] = []
        yaxis_srt[yaxis_info][xaxis_info].append(f)
    
    # show discovered variables
    yaxis_disc = list(yaxis_srt.keys())
    yaxis_disc = sorted(yaxis_disc)
    xaxis_disc = sorted(xaxis_disc)
    # special axis orderings
    yaxis_disc = special_order(yaxis_disc, yaxis)
    xaxis_disc = special_order(xaxis_disc, xaxis)
    print('yaxis discovered settings:')
    print(yaxis_disc)
    print('xaxis discovered settings:')
    print(xaxis_disc)
    print('-')
    print('all discovered settings:')
    for v in valid_variables:
        full_discovery[v] = sorted(full_discovery[v])
        print(v)
        print(full_discovery[v])

    # check or set fixed variables
    print('-')
    print('SUMMARY OF GRID SETTINGS:')
    cmd_settings = {}
    cmd_settings['pos'] = pos
    cmd_settings['blk'] = blk
    cmd_settings['head'] = head
    cmd_settings['img'] = img
    all_variable_ops = [0]
    for v in valid_variables:
        print(v)
        if v == xaxis:
            print('  xaxis variable')
            full_discovery[v]
            cmd_settings[v] = None
        elif v == yaxis:
            print('  yaxis variable')
            full_discovery[v]
            cmd_settings[v] = None
        elif cmd_settings[v] is not None:
            print('  cmd setting: %s'%cmd_settings[v])
            if cmd_settings[v] not in full_discovery[v]:
                print('ERROR: did not find any images with this setting')
                exit(-1)
        else:
            if use_all is not None and use_all == v:
                all_variable_ops = full_discovery[v]
                print('  Using all found options for variable %s'%v)
                print(full_discovery[v])
                # print('  found %i options'%len(all_variable_ops))
            else:
                print('  cmd setting not given. selecting one option from those discovered:')
                cmd_settings[v] = full_discovery[v][0]
                print('  ' + cmd_settings[v])

    # iterate over 'all' variable, if use_all set:
    for avo in all_variable_ops:
        if use_all is not None:
            cmd_settings[use_all] = avo
        # generate grid
        rows = []
        if xaxis == 'img': # images on top
            row = []
            for f in full_discovery['img']:
                fn = os.path.join(input_dir, f)
                image = load_image(fn, img_size)
                row.append(image)
            row = concat_with_pad(row, 1, pad)
            rows.append(row)
        for y in yaxis_disc:
            row = []
            if yaxis == 'img': # images on left side
                fn = os.path.join(input_dir, y)
                image = load_image(fn, img_size)
                row.append(image)
            for x in xaxis_disc:
                # select the right image:
                sel_img = None
                for f in yaxis_srt[y][x]:
                    f_info = parse_filename(f)
                    valid_match = True
                    for v in valid_variables:
                        if cmd_settings[v] is not None and cmd_settings[v] != f_info[v]:
                            valid_match = False
                    if valid_match:
                        sel_img = f
                        break
                if sel_img is None:
                    print('ERROR: failed to find image for:')
                    print('%s = %s'%(xaxis, x))
                    print('%s = %s'%(yaxis, y))
                    exit(-1)
                fn = os.path.join(input_dir, sel_img)
                if resize_needed:
                    image = load_image(fn, img_size)
                else:
                    image = load_image(fn)
                row.append(image)
            row = concat_with_pad(row, 1, pad)
            rows.append(row)
        supergrid = concat_with_pad(rows, 0, pad)
        # Save full size and small versions
        os.makedirs(output_dir, exist_ok = True)
        if mod_id is None:
            fname = ''
        else:
            fname = mod_id + '_'
        for v in valid_variables:
            if xaxis == v:
                fset = 'XAXIS'
            elif yaxis == v:
                fset = 'YAXIS'
            else:
                fset = cmd_settings[v]
            fname += '(%s %s)_'%(v, fset)
        fname = fname[:-1]
        fname_small = 'small_' + fname
        fname = os.path.join(output_dir, fname) + '.png'
        fname_small = os.path.join(output_dir, fname_small) + '.png'
        print('saving: ' + fname)
        h = supergrid.shape[0]
        w = supergrid.shape[1]
        supergrid = Image.fromarray(supergrid)
        supergrid.save(fname)
        if save_small:
            supergrid_small = supergrid.resize([int(w * scale_factor), int(h * scale_factor)])
            supergrid_small.save(fname_small)



########################################



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Make grid of Visualized Self-Attention maps')
    # locations
    parser.add_argument("input_dir", type=str, help="Path of dir with images")
    parser.add_argument("--output_dir", default="supergrids", type=str, help="Path of output dir")
    # grid settings
    parser.add_argument("--xaxis", default="pos", type=str, help="which variable to change on the xaxis")
    parser.add_argument("--yaxis", default="blk", type=str, help="which variable to change on the yaxis")
    parser.add_argument("--all", default=None, type=str, help="optional, apply every found option for a 3rd variable")
    # fixed variable settings
    parser.add_argument("--pos", default=None, type=str, help='fixed position, if not an axis variable')
    parser.add_argument("--blk", default=None, type=str, help='fixed block, if not an axis variable')
    parser.add_argument("--head", default=None, type=str, help='fixed head, if not an axis variable')
    parser.add_argument("--img", default=None, type=str, help='fixed image, if not an axis variable')
    # other
    parser.add_argument("--img_size", default=480, type=int, help="Resize images, if img is an axis variable.")
    parser.add_argument("--pad", default=10, type=int, help='padding between images')
    parser.add_argument("--scale_factor", default=0.125, type=float, help='scale factor for small version of image')
    args = parser.parse_args()
    make_supergrid(args.input_dir, args.output_dir, args.xaxis, args.yaxis, args.all, args.pos, args.blk, args.head,
        args.img, args.img_size, args.pad, args.scale_factor)