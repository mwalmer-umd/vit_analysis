def get_model_wrapper(meta_model, arch, patch, imsize, extract_mode='both', blk_sel='all'):
    if meta_model == 'dino':
        from wrapper_dino import DINO_Wrapper
        return DINO_Wrapper(arch, patch, imsize, extract_mode, blk_sel)
    if meta_model == 'clip':
        from wrapper_clip import CLIP_Wrapper
        return CLIP_Wrapper(arch, patch, imsize, extract_mode, blk_sel)
    if meta_model == 'mae':
        from wrapper_mae import MAE_Wrapper
        return MAE_Wrapper(arch, patch, imsize, extract_mode, blk_sel)
    if meta_model == 'timm':
        from wrapper_timm import TIMM_Wrapper
        return TIMM_Wrapper(arch, patch, imsize, extract_mode, blk_sel)
    if meta_model == 'moco':
        from wrapper_moco import MOCO_Wrapper
        return MOCO_Wrapper(arch, patch, imsize, extract_mode, blk_sel)
    if meta_model == 'beit':
        from wrapper_beit import BEIT_Wrapper
        return BEIT_Wrapper(arch, patch, imsize, extract_mode, blk_sel)
    if meta_model == 'random':
        from wrapper_random import Random_Wrapper
        return Random_Wrapper(arch, patch, imsize, extract_mode, blk_sel)
    print('ERROR: Unknown Meta Model: ' + meta_model)
    exit(-1)