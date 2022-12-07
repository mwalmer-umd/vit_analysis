# image pre-processing utils
from torchvision import transforms

MEAN_VALS = {
    'dino' : [0.485, 0.456, 0.406],
    'clip' : [0.48145466, 0.4578275, 0.40821073],
    'mae'  : [0.485, 0.456, 0.406],
    'timm' : [0.5000, 0.5000, 0.5000],
    'moco' : [0.485, 0.456, 0.406],
    'beit' : [0.5000, 0.5000, 0.5000],
    'random' : [0.5000, 0.5000, 0.5000],
}

STD_VALS = {
    'dino' : [0.229, 0.224, 0.225],
    'clip' : [0.26862954, 0.26130258, 0.27577711],
    'mae'  : [0.229, 0.224, 0.225],
    'timm' : [0.5000, 0.5000, 0.5000],
    'moco' : [0.229, 0.224, 0.225],
    'beit' : [0.5000, 0.5000, 0.5000],
    'random' : [0.5000, 0.5000, 0.5000],
}



'''
get a stanardardized pre-processing transformation
for all models. Image resizing and cropping is kept
consistent, while the normalization values are kept
from each model's original pre-processing
'''
def standard_transform(meta_model, imsize):
    if meta_model not in MEAN_VALS:
        print('ERROR: standard_tranform does not support: %s'%meta_model)
        exit(-1)        
    transform = transforms.Compose([
        transforms.Resize(imsize, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
        transforms.CenterCrop(imsize),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_VALS[meta_model], STD_VALS[meta_model]),
    ])
    return transform



# minimal transform for use with dense feature extraction
def minimal_transform(meta_model):
    if meta_model not in MEAN_VALS:
        print('ERROR: standard_tranform does not support: %s'%meta_model)
        exit(-1)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_VALS[meta_model], STD_VALS[meta_model]),
    ])
    return transform