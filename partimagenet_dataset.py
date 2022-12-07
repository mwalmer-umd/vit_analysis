import os
import json

from pycocotools.coco import COCO
from PIL import Image


'''
Will sample 'per_super_class'-many samples from each superclass, evenly distributed between
the subclasses, except when a subclass run out, then it will continue sampling from the others
'''
class PartImagenetDataset():
    def __init__(self, dataroot='data/PartImageNet', per_super_class=100, show_info=False):
        self.dataroot = dataroot
        self.parts = ['train', 'val', 'test']
        self.per_super_class = per_super_class
        # load coco annotations for all 3 parts, and make a dictionary mapping file names to their part and id
        self.cocos = {}
        self.name2partandid = {}
        for p in self.parts:
            coco = COCO(os.path.join(self.dataroot, '%s.json'%p))
            self.cocos[p] = coco
            ids = list(sorted(coco.imgs.keys()))
            for im_id in ids:
                img_info = coco.loadImgs(im_id)[0]
                self.name2partandid[img_info['file_name']] = (p, im_id)
        # for compatibility functions, save one coco instance as self.coco
        self.coco = self.cocos['val']
        # imagenet label name
        fn = os.path.join(self.dataroot, 'imagenet_class_index.json')
        with open(fn, 'r') as f:
            data = json.load(f)
        self.imnet_id_dict = {}
        for d in data.keys():
            self.imnet_id_dict[data[d][0]] = data[d][1]
        # examine all 3 partitions, and gather labels by partion
        self.imnet_labels = {}
        for p in self.parts:
            partroot = os.path.join(self.dataroot, p)
            labels = os.listdir(partroot)
            for l in labels:
                if l == '.DS_Store': continue
                # examine sample images to identify super category
                labroot = os.path.join(partroot, l)
                images = os.listdir(labroot)
                if '.DS_Store' in images:
                    images.remove('.DS_Store')
                supercategory = None
                for im in images:
                    im_p, im_id = self.name2partandid[im]
                    im_anns = self.cocos[im_p].loadAnns(self.cocos[im_p].getAnnIds(im_id))
                    for a in im_anns:
                        cat = a['category_id']
                        cat_info = self.cocos[im_p].loadCats(cat)[0]
                        supercategory = cat_info['supercategory']
                        break
                    if supercategory is not None: break
                label_info = {}
                label_info['part'] = p
                label_info['name'] = self.imnet_id_dict[l]
                label_info['supercategory'] = supercategory
                label_info['count'] = len(images)
                self.imnet_labels[l] = label_info
        # organize super categories
        self.supercats = {}
        for l in self.imnet_labels:
            sc = self.imnet_labels[l]['supercategory']
            if sc not in self.supercats:
                sc_info = {}
                sc_info['imnet_labels'] = []
                sc_info['count'] = 0
                self.supercats[sc] = sc_info
            self.supercats[sc]['imnet_labels'].append(l)
            self.supercats[sc]['count'] += self.imnet_labels[l]['count']
        # info
        if show_info:
            print('=============')
            print('DATASET INFO:')
            for sc in self.supercats:
                sc_info = self.supercats[sc]
                print('%s - %i'%(sc, sc_info['count']))
                for l in sc_info['imnet_labels']:
                    label_info = self.imnet_labels[l]
                    print('  %s (%s) - %i'%(l, label_info['name'], label_info['count']))
        # select which images to sample
        self.samples = {}
        for sc in self.supercats:
            sc_info = self.supercats[sc]
            if per_super_class > sc_info['count']:
                print('WARNING: Superclass %s has <%i total samples (%i)'%(sc, per_super_class, self.supercats[sc]['count']))
                for l in sc_info['imnet_labels']:
                    l_info = self.imnet_labels[l]
                    labroot = os.path.join(self.dataroot, l_info['part'], l)
                    images = os.listdir(labroot)
                    if '.DS_Store' in images:
                        images.remove('.DS_Store')
                    self.samples[l] = images
            else:
                # iteratively sample from classes until count reached
                sc_count = 0
                active_labels = sc_info['imnet_labels']
                active_lists = {}
                active_i = 0
                warn_list = []
                for l in active_labels:
                    self.samples[l] = []
                    l_info = self.imnet_labels[l]
                    labroot = os.path.join(self.dataroot, l_info['part'], l)
                    images = os.listdir(labroot)
                    images.sort()
                    if '.DS_Store' in images:
                        images.remove('.DS_Store')
                    active_lists[l] = images
                while sc_count < per_super_class:
                    l = active_labels[active_i]
                    active_i = (active_i + 1)%len(active_labels)
                    if len(active_lists[l]) == 0:
                        if l not in warn_list:
                            print('WARNING: Class %s (%s) has insufficient samples'%(l, self.imnet_labels[l]['name']))
                            warn_list.append(l)
                        continue
                    im = active_lists[l][0]
                    active_lists[l].pop(0)
                    self.samples[l].append(im)
                    sc_count += 1
        # info on samples
        if show_info:
            print('=============')
            print('SAMPLE INFO:')
            for sc in self.supercats:
                sc_info = self.supercats[sc]
                print(sc)
                sc_count = 0
                for l in sc_info['imnet_labels']:
                    label_info = self.imnet_labels[l]
                    sample_count = len(self.samples[l])
                    sc_count += sample_count
                    print('  %s (%s) - %i'%(l, label_info['name'], sample_count))
                print('  [TOTAL: %i]'%sc_count)
        # gather selected images into a single list
        self.all_samples = []
        for l in self.samples:
            self.all_samples += self.samples[l]
        if show_info:
            print('[TOTAL SAMPLES]')
            print(len(self.all_samples))
            self.count_annotations()


    # helper to display the number of annotations for the selected samples
    def count_annotations(self):
        anno_counts = {}
        for im in self.all_samples:
            im_p, im_id = self.name2partandid[im]
            im_anns = self.cocos[im_p].loadAnns(self.cocos[im_p].getAnnIds(im_id))
            for a in im_anns:
                cat = a['category_id']
                cat_info = self.cocos[im_p].loadCats(cat)[0]
                cat_name = cat_info['name']
                if cat_name not in anno_counts:
                    anno_counts[cat_name] = 0
                anno_counts[cat_name] += 1
        print('[ANNOTATION COUNTS:]')
        anno_tot = 0
        for a in sorted(list(anno_counts.keys())):
            print('  %s - %i'%(a, anno_counts[a]))
            anno_tot += anno_counts[a]
        print(  '[TOTAL: %i]'%anno_tot)



    def _load_image(self, img_name):
        im_p, _ = self.name2partandid[img_name]
        im_l = img_name.split('_')[0]
        path = os.path.join(self.dataroot, im_p, im_l, img_name)
        return Image.open(path).convert("RGB")



    def _load_target(self, img_name):
        im_p, im_id = self.name2partandid[img_name]
        # for compatibility functions, set self.coco to the most recently used coco
        self.coco = self.cocos[im_p]
        return self.cocos[im_p].loadAnns(self.cocos[im_p].getAnnIds(im_id))



    def __getitem__(self, index):
        img_name = self.all_samples[index]
        image = self._load_image(img_name)
        target = self._load_target(img_name)
        return image, target



    def __len__(self):
        return len(self.all_samples)



############################## HELPERS ##############################



# check sampling
def main(args):
    dataset = PartImagenetDataset(args.dataroot, args.persc, show_info=True)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('PartImageNet Check Sampling')
    parser.add_argument('--dataroot', default='data/PartImageNet')
    parser.add_argument('--persc', type=int, default=100, help='check sampling for number of samples per superclass')
    args = parser.parse_args()
    main(args)