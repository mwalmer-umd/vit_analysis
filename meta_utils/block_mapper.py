# meta utility that maps arch + block selection to index, with handling for
# special block names: all, first, q1, middle, q3, last

DEPTH_MAP = {
    'S' : 12,
    'B' : 12,
    'L' : 24,
    'H' : 32,
}

def block_mapper(arch, blk_sel):
    assert arch in DEPTH_MAP
    special_options = ['all', 'first', 'q1', 'middle', 'q3', 'last']
    # handle variable input blk_sel as item or list
    if not isinstance(blk_sel, list):
        blk_sel = [blk_sel]
    # all overrides other options
    if 'all' in blk_sel:
        blk_sel = ['all']
    idx_sel = []
    for b in blk_sel:
        try:
            # integer selection (string or int)
            bi = int(b)
            if bi < 0 or bi >= DEPTH_MAP[arch]:
                print('ERROR: invalid block selection %i for ViT-%s'%(bi, arch))
                exit(-1)
            idx_sel.append(bi)
        except:
            # special option                
            if b == 'all':
                for i in range(DEPTH_MAP[arch]):
                    idx_sel.append(i)
            elif b == 'first':
                idx_sel.append(0)
            elif b == 'q1':
                idx_sel.append(int(DEPTH_MAP[arch]/4) - 1)
            elif b == 'middle':
                idx_sel.append(int(DEPTH_MAP[arch]/2) - 1)
            elif b == 'q3':
                idx_sel.append(int(3*DEPTH_MAP[arch]/4) - 1)
            elif b == 'last':
                idx_sel.append(DEPTH_MAP[arch] - 1)
            else:
                print('ERROR: unknown block option: %s'%b)
                exit(-1)
    idx_sel = sorted(idx_sel)
    # test for duplicates
    ret = []
    for idx in idx_sel:
        if idx in ret:
            print('WARNING: block selection included redundant selection for block %i'%idx)
        else:
            ret.append(idx)   
    return ret