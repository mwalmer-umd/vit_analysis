"""
###########################################################################
Collection of metrics for transformers, designed to be computed
incrementally over batches

Written by: Matthew Walmer
###########################################################################
"""
import matplotlib.pyplot as plt
import torch
import numpy as np

from cka.utils import cluster_metric


########################################



# track the average attention distance per head, based on the attention distance metric proposed in
# "Do Vision Transformers See Like Convolutional Neural Networks" (Raghu et al.)
class AttentionDistance():
    def __init__(self, debug=False):
        self.metric_name = 'avg-att-dist'
        self.results = []
        self.distance_template = None
        self.edge_len = None
        self.debug = debug
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # treat tokens as existing on a [0,1] square. convert their
    # integer position to a spatial position
    def convert_to_xy(self, i):
        x = i % self.edge_len
        y = int((i-x)/self.edge_len)
        x /= (self.edge_len - 1)
        y /= (self.edge_len - 1)
        return x, y

    # pre-calculate distanced between all token pairs treat tokens as existing on a [0,1] 
    # square, requires a square patch grid
    def prep_distance_template(self, attentions):
        if self.distance_template is not None:
            return
        print('Preparing Distance Template')
        att = attentions[:,:,:,1:,1:] # Remove CLS token
        self.edge_len = int(np.sqrt(att.shape[3]))
        if self.edge_len * self.edge_len != att.shape[3] or self.edge_len * self.edge_len != att.shape[4]:
            print('ERROR: attention distance requires square token layout')
            exit(-1)
        nt = att.shape[3]
        convert_template = np.zeros([nt, 2])
        for i in range(nt):
            x, y = self.convert_to_xy(i)
            convert_template[i,:] = [x, y]
        distance_template = torch.zeros([nt, nt])
        for i in range(nt):
            xy = convert_template[i,:]
            d = convert_template - xy
            d = np.square(d)
            d = np.sum(d, axis=1)
            d = np.sqrt(d)
            distance_template[i,:] = torch.from_numpy(d)
        if self.debug: # visualize distance maps
            pos_sel = [0, 30, 1770, 3570, 3599]
            for p in pos_sel:
                dis = distance_template[p,:]
                dis = dis.reshape(edge_len, edge_len).cpu().numpy()
                fname = 'debug_%i.png'%p
                plt.imsave(fname=fname, arr=dis, format='png')
        self.distance_template = distance_template.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)

    def add(self, attentions, labels):
        self.prep_distance_template(attentions)
        att = attentions[:,:,:,1:,1:] # Remove CLS token
        att_sum = torch.sum(att, dim=4, keepdim=True)
        # handle empty case (all att on cls token)
        att_nul = (att_sum == 0).to(torch.long)
        att_sum += att_nul
        att = att / att_sum
        d = torch.sum(att * self.distance_template, dim=4)
        d = torch.mean(d, dim=3)
        self.results.append(d.cpu().numpy())

    def get_results(self):
        return np.concatenate(self.results, axis=0)



########################################



class PositionDeviation():
    def __init__(self):
        self.metric_name = 'stdev-over-token-pos'
        self.results = []

    def add(self, attentions, labels):
        att = attentions[:,:,:,1:,1:] # Remove CLS token
        d = torch.std(att, dim=3)
        d = torch.mean(d, dim=3)
        self.results.append(d.cpu().numpy())

    def get_results(self):
        return np.concatenate(self.results, axis=0)



########################################



class SpatialCLSAttention():
    def __init__(self):
        self.metric_name = 'spatial-cls-att'
        self.results = []

    def add(self, attentions, labels):
        att = attentions[:,:,:,1:,1] # Remove CLS token, and observe only CLS attention pos
        v = torch.mean(att, dim=3)
        self.results.append(v.cpu().numpy())

    def get_results(self):
        return np.concatenate(self.results, axis=0)



########################################



# determine if the sparse attention patterns in the late layers of CLIP and TIMM
# encode semantic "shortcuts" by using the patterns directly as features and running
# semantic cluster purity analysis. This metric is still experimental
class AttentionPatternShortcuts():
    def __init__(self):
        self.metric_name = 'att-pat-shortcuts'
        self.att_pats = []
        self.labs = []

    def add(self, attentions, labels):
        bs = attentions.shape[0]
        for i in range(bs):
            # att = attentions[i,:,:,1:,1:] # remove CLS token
            att = attentions[i,...] # dont't remove CLS token
            att = torch.mean(att, dim=2) # average spatial source potions
            # option 1: focus on a single head to test the hypothesis
            # att = att[-2, 0, :]
            # option 2: average per-block
            att = att[-2, :, :]
            att = torch.mean(att, dim=0)
            # store attention patterns and labels
            self.att_pats.append(att.cpu().numpy())
            self.labs.append(labels[i].cpu())

    def get_results(self):
        att_pats = np.stack(self.att_pats, axis=0)
        all_labs = np.array(self.labs)
        print(att_pats.shape)
        print(all_labs.shape)
        print('running clustering analysis')
        nmi, ari, pur = cluster_metric(50, att_pats, all_labs)
        exit()

        return None
        # return np.concatenate(self.results, axis=0)



########################################


'''
average amount of attention ON each token 
also equivalent to averaging all attention maps over
the source-token dimension. Runs in 4 modes:
    all - average attention of all source tokens
    cls - take only the cls token
    spc - take only the spatial tokens
    spcpure - take only the spatial tokens and remove
        cls token as a destination token (normalize
        for the lost attention mass)

aggregated maps include the cls token in the
set of destination-tokens, which can then be
included or excluded in further plots
'''
class AvgAttentionOnToken():
    def __init__(self, mode='all'):
        assert mode in ['all', 'cls', 'spc', 'spcpure']
        if mode == 'all':
            self.metric_name = 'avg-att-on-token'
        else:
            self.metric_name = 'avg-%s-att-on-token'%mode
        self.mode = mode
        self.average_acc = None
        self.count = 0

    def add(self, attentions, labels):
        self.count += attentions.shape[0]
        if self.mode == 'all':
            attentions = torch.mean(attentions, dim=3)
        elif self.mode == 'spc':
            attentions = attentions[:,:,:,1:,:]
            attentions = torch.mean(attentions, dim=3)
        elif self.mode == 'spcpure':
            attentions = attentions[:,:,:,1:,:] # remove source-cls-token
            attentions[:,:,:,:,0] = 0 # remove all attention on CLS tokens
            att_sum = torch.sum(attentions, dim=4, keepdim=True)
            # handle and track empty case (all att on cls token)
            att_nul = (att_sum == 0).to(torch.long)
            att_sum += att_nul
            attentions = attentions / att_sum # normalize for removed attention mass
            attentions = torch.mean(attentions, dim=3)
        else: # cls
            attentions = attentions[:,:,:,0,:]
        avg_att = torch.sum(attentions, dim=0)
        avg_att = avg_att.cpu().numpy()
        if self.average_acc is None:
            self.average_acc = avg_att
        else:
            self.average_acc += avg_att

    def get_results(self):
        return self.average_acc / self.count



########################################



'''
Measure the average attention positions of all spatial tokens, but now
the heat maps  are shifted so the current token is always aligned with
the center. CLS token is removed and the maps are normalized to account
for the lost attention mass.

Closest match in AvgAttentionOnToken would be spcpure mode, since this
method remove the cls token completely and normalizes for the lost mass

Tf the original token grid is KxK, the output of this method will be
(2K-1)x(2K-1).
'''
class AvgAlignedAttentionOnToken():
    def __init__(self):
        self.metric_name = 'avg-aligned-att-on-token'
        self.edge_len = None
        self.c = None
        self.average_acc = None
        self.count = 0
        # track null events (all attention on CLS token)
        self.null_count = 0
        self.null_imgc = 0
        self.null_tracker = None


    # convert direct token idx to i,j coordinates (i=down, j=across)
    def convert_to_ij(self, idx):
        j = idx % self.edge_len
        i = int((idx-j)/self.edge_len)
        return i, j

    def add(self, attentions, labels):
        # Remove CLS token from source and destination, and normalize for lost attention mass
        att = attentions[:,:,:,1:,1:] 
        att_sum = torch.sum(att, dim=4, keepdim=True)
        # handle and track empty case (all att on cls token)
        att_nul = (att_sum == 0).to(torch.long)
        nul_img = (torch.sum(att_nul, dim=(1,2,3,4)) > 0)
        self.null_count += torch.sum(att_nul)
        self.null_imgc += torch.sum(nul_img)
        if self.null_tracker is None:
            self.null_tracker = torch.sum(att_nul, dim=(0,4))
        else:
            self.null_tracker += torch.sum(att_nul, dim=(0,4))
        att_sum += att_nul
        # normalize for removed attention mass
        att = att / att_sum
        # square-ify the token grid
        if self.edge_len is None:
            self.edge_len = int(np.sqrt(att.shape[3]))
            if self.edge_len * self.edge_len != att.shape[3] or self.edge_len * self.edge_len != att.shape[4]:
                print('ERROR: attention distance requires square token layout')
                exit(-1)
        att_sq = torch.reshape(att, [att.shape[0], att.shape[1], att.shape[2], att.shape[3], self.edge_len, self.edge_len]).cpu().numpy()
        # initialize attention maps with padding
        if self.average_acc is None:
            w = (self.edge_len * 2) - 1
            self.average_acc = np.zeros([att.shape[1], att.shape[2],  w, w], dtype=att_sq.dtype)
        # accumulate
        for idx in range(att.shape[3]):
            cur = att_sq[:, :, :, idx, :, :]
            cur = np.sum(cur, axis=0)
            i, j = self.convert_to_ij(idx)
            e = self.edge_len
            self.average_acc[:,:,e-i-1:e-i-1+e,e-j-1:e-j-1+e] += cur
            self.count += 1

    def get_results(self):
        if self.null_count > 0:
            print('Null Report:')
            print('avg-aligned-att-on-token')
            print('null count: %i'%self.null_count)
            print('null images: %i'%self.null_imgc)
            print('null tokens: %i'%torch.sum(self.null_tracker > 0))
            print('null heads:')
            print(torch.sum(self.null_tracker, dim=2))
        return self.average_acc / self.count



########################################



# average amount of attention ON each token separated by image class, to test if some heads/tokens
# are sensitive to particular object classes
class AvgAttentionOnTokenPerClass():
    def __init__(self, num_class=50):
        self.metric_name = 'avg-att-on-token-per-class'
        self.average_accs = {}
        self.counts = {}
        self.nc = num_class
        for i in range(self.nc):
            self.average_accs[i] = None
            self.counts[i] = 0

    def add(self, attentions, labels):
        b = attentions.shape[0]
        for i in range(b):
            l = int(labels[i])
            avg_att = attentions[i,...]
            avg_att = torch.mean(avg_att, dim=2)
            avg_att = avg_att.cpu().numpy()
            if self.average_accs[l] is None:
                self.average_accs[l] = avg_att
            else:
                self.average_accs[l] += avg_att
            self.counts[l] += 1

    def get_results(self):
        temp = self.average_accs[0]
        nb = temp.shape[0]
        nh = temp.shape[1]
        nt = temp.shape[2]
        ret = np.zeros([self.nc, nb, nh, nt], dtype=temp.dtype)
        for i in range(self.nc):
            ret[i,...] = self.average_accs[i] / self.counts[i]
        return ret



########################################



# measure standard deviation over heads, a per-block metric
class HeadDeviation():
    def __init__(self):
        self.metric_name = 'stdev-over-head'
        self.results = []

    def add(self, attentions, labels):
        att = attentions[:,:,:,1:,1:] # Remove CLS token
        d = torch.std(att, dim=2)
        d = torch.mean(d, dim=2)
        d = torch.mean(d, dim=2)
        self.results.append(d.cpu().numpy())

    def get_results(self):
        return np.concatenate(self.results, axis=0)



########################################
# POST PROCESSING METRICS
########################################
# Metrics that are derived by directly post-processing the results of other metrics
# typically reducing lower-level metrics to higher-level metrics



# take the per-class average attention maps and compute the average stdev over class
# input is the result of AvgAttentionOnTokenPerClass
def deviation_by_class(result):
    res = np.std(result, axis=0, keepdims=True)
    res = np.mean(res, axis=3)
    return res



# take in the result for 'avg-aligned-att-on-token' and compute the average center
# of attention for each block and the average offset distance
def average_att_offset(result):
    nb = result.shape[0]
    nh = result.shape[1]
    nt = result.shape[2]
    # distance templates
    dtx = np.zeros([nt, nt])
    dty = np.zeros([nt, nt])
    for i in range(nt):
        v = 2 * (float(i)/(nt-1)) - 1
        dtx[:,i] = v
        dty[i,:] = v
    # compute average centers
    cx = np.zeros([nb, nh])
    cy = np.zeros([nb, nh])
    d = np.zeros([nb, nh])
    for b in range(nb):
        for h in range(nh):
            # make sure heat map sums to 1
            # special case: if attention is fully on CLS token, cannot divide by zero
            att = result[b,h,...]
            div = np.sum(att)
            if div > 0:
                att /= np.sum(att)
            cxbh = np.sum(np.multiply(att, dtx))
            cybh = np.sum(np.multiply(att, dty))
            d[b,h] = np.linalg.norm(np.array([cxbh, cybh]))
            cx[b,h] = cxbh
            cy[b,h] = cybh
    return d, cx, cy