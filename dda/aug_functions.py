from locale import CODESET
from autoalbument.albumentations_pytorch import functional as autoF
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta
import torch.multiprocessing as mp
from torchvision.transforms import RandomErasing
import random
from itertools import combinations, chain
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
def self_mix(data, value=None, min_v=None, max_v=None):
    # value = (max_v - min_v) * value + min_v
    size = data.size()
    W = size[-1]
    H = size[-2]
    ## uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    cut_w = W // 2
    cut_h = H // 2

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    while True:
        bbxn = np.random.randint(0, W-(bbx2-bbx1))
        bbyn = np.random.randint(0, H-(bby2-bby1))

        if bbxn != bbx1 or bbyn != bby1:
            break

    ## random rotate the croped when it's square
    if (bbx2 - bbx1) == (bby2 - bby1):
        k = random.sample([0, 1, 2, 3], 1)[0]
    else:
        k = 0
    x = data.clone()

    x[:, :, bbx1:bbx2, bby1:bby2] = torch.rot90(x[:, :, bbxn:bbxn + (bbx2-bbx1), bbyn:bbyn + (bby2-bby1)], k, [2,3])
    return x + data - data.detach()

def self_mix_set(data, value=None, min_v=None, max_v=None, set_sel='S', supp_num=None):
    x = self_mix(data, value, min_v, max_v)
    dshape = data.shape
    if set_sel == 'Q':
        mask = torch.cat([torch.zeros_like(data)[0:supp_num],torch.ones_like(data)[supp_num:]],0)
    elif set_sel == 'S':
        mask = torch.cat([torch.ones_like(data)[0:supp_num],torch.zeros_like(data)[supp_num:]],0)
    x = mask * x + (1-mask) * data
    return x

def random_erase(data, value=None, scale=(.02, .4), ratio=(.3, 1/.3)):
    x = RandomErasing(scale=scale, ratio=ratio)(data.clone())
    return x + data - data.detach() 

def random_erase_set(data, value=None, min_v=None, max_v=None, set_sel='S', supp_num=None):
    x = random_erase(data, value)
    if set_sel == 'Q':
        mask = torch.cat([torch.zeros_like(data)[0:supp_num],torch.ones_like(data)[supp_num:]],0)
    elif set_sel == 'S':
        mask = torch.cat([torch.ones_like(data)[0:supp_num],torch.zeros_like(data)[supp_num:]],0)
    x = mask * x + (1-mask) * data
    return x

def class_rotation(data, value, n_classes, batch_size, rotate90_times=[0, 0, 0, 1, 2, 3]):
    img_shape = data.shape[-3:]
    x = data.reshape(batch_size, n_classes, -1, *img_shape).clone()
    x = x.permute(1,0,2,3,4,5)
    for j in range(n_classes):
        k = random.sample(rotate90_times, 1)[0]
        x[j] = torch.rot90(x[j], k, [3, 4]) 
    x = x.permute(1,0,2,3,4,5)
    x = x.reshape(-1, *img_shape)
    return x + data - data.detach()





def shift_r(x, value, min_v, max_v):
    value = (max_v - min_v) * value + min_v
    sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
    value = value * sign
    return autoF.shift_rgb(
            x,
            r_shift=value,
            g_shift=0.0,
            b_shift=0.0
        )

def shift_g(x, value, min_v, max_v):
    value = (max_v - min_v) * value + min_v
    sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
    value = value * sign
    return autoF.shift_rgb(
            x,
            r_shift=0.0,
            g_shift=value,
            b_shift=0.0
        )
def shift_b(x, value, min_v, max_v):
    value = (max_v - min_v) * value + min_v
    sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
    value = value * sign
    return autoF.shift_rgb(
            x,
            r_shift=0.0,
            g_shift=0.0,
            b_shift=value
        )

def random_brightness(x, value, min_v, max_v):
    value = (max_v - min_v) * value + min_v
    sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
    value = value * sign
    return autoF.brightness_adjust(x, beta=value)

def random_contrast(x, value, min_v, max_v):
    neg_scale = -(1.0-min_v)
    pos_scale = max_v - 1.0
    sign = pos_scale if torch.rand(1).item() > 0.5 else neg_scale
    value = value * sign + 1.0
    return autoF.contrast_adjust(x, alpha=value)

def solarize(x, value, min_v, max_v):
    value = max_v - (max_v - min_v) * value
    return autoF.solarize(x, threshold=value) + value - value.detach() + x - x.detach()

def hflip(x, value, min_v, max_v):
    return autoF.hflip(x)

def vflip(x, value, min_v, max_v):
    return autoF.vflip(x)

def rotate(x, value, min_v, max_v):
    value = (max_v - min_v) * value + min_v
    sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
    value = value * sign
    return autoF.rotate(x, angle=value)

def shift_x(x, value, min_v, max_v):
    value = (max_v - min_v) * value + min_v
    sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
    value = value * sign
    return autoF.shift_x(x, dx=value)

def shift_y(x, value, min_v, max_v):
    value = (max_v - min_v) * value + min_v
    sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
    value = value * sign
    return autoF.shift_y(x, dy=value)

def scale(x, value, min_v, max_v):
    neg_scale = -(1.0-min_v)
    pos_scale = max_v - 1.0
    sign = pos_scale if torch.rand(1).item() > 0.5 else neg_scale
    value = value * sign + 1.0
    return autoF.scale(x, scale=value)

def posterize(x, value, min_v, max_v):
    # mag: 0 to 1
    value = (max_v - min_v) * value + min_v
    value = value.view(-1, 1, 1, 1)
    with torch.no_grad():
        shift = (value * 8).long()
        shifted = (x.mul(255).long() << shift) >> shift
    return shifted.float() / 255 + value - value.detach()

def equalize(img, value, min_v, max_v):
    # see https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py#L319
    with torch.no_grad():
        # BCxHxW
        reshaped = img.clone().flatten(0, 1).clamp_(0, 1) * 255
        size = reshaped.size(0)  # BC
        # 0th channel [0-255], 1st channel [256-511], 2nd channel [512-767]...(BC-1)th channel
        shifted = reshaped + 256 * torch.arange(0, size, device=reshaped.device,
                                                dtype=reshaped.dtype).view(-1, 1, 1)
        # channel wise histogram: BCx256
        histogram = shifted.histc(size * 256, 0, size * 256 - 1).view(size, 256)
        # channel wise cdf: BCx256
        cdf = histogram.cumsum(-1)
        # BCx1
        step = ((cdf[:, -1] - histogram[:, -1]) / 255).floor_().view(size, 1)
        # cdf interpolation, BCx256
        cdf = torch.cat([cdf.new_zeros((cdf.size(0), 1)), cdf], dim=1)[:, :256] + (step / 2).floor_()
        # to avoid zero-div, add 0.1
        output = (cdf / (step + 0.1)).floor_().view(-1)[shifted.long()].reshape_as(img) / 255
    return output+img-img.detach()

def cutout_fixed_num_holes(x, value, num_holes=16, image_shape=(84,84)):
    height, width = image_shape
    min_size = min(height, width)
    hole_size = max(int(min_size * value), 0)
    return autoF.cutout(x, num_holes=num_holes, hole_size=hole_size) - value.detach() + value + x - x.detach()
     
def cutout_fixed_size(x, value, min_v, max_v, hole_size_divider=16, image_shape=(84,84)):
    value = (max_v - min_v) * value + min_v
    height, width = image_shape
    min_size = min(height, width)
    hole_size = int(min_size // hole_size_divider)
    hole_size = max(hole_size, 1)
    return autoF.cutout(x, num_holes=int(value), hole_size=hole_size) - value.detach() + value + x - x.detach()

def identity(x,value, min_v, max_v):
    return x

def sample_pairing(x,value, min_v, max_v):
    value = (max_v - min_v) * value + min_v
    indices = torch.randperm(x.size(0), device=x.device, dtype=torch.long)
    value = value.view(-1, 1, 1, 1)
    return (1 - value) * x + value * x[indices]

def rand_bbox(size, lam=0.5):
    W = size[-1]
    H = size[-2]
    cut_rat = np.sqrt(lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    ## uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def qcutmix(data, value, batch_size, supp_num):
    with torch.no_grad():
        labels = data.label[:,supp_num:]
        N, C, H, W = data.shape
        query_num =  N//batch_size - supp_num
        # mask = torch.zeros_like(data).reshape(batch_size, N//batch_size, C, H, W)
        # mask[:,supp_num:] = 1.0
        X = data.reshape(batch_size,N//batch_size,C, H, W)
        qX = X[:,supp_num:] #query data
        sX = X[:,0:supp_num]
        label_a, label_b = torch.zeros_like(labels), torch.zeros_like(labels)
        new_data = torch.zeros_like(qX)
        b = len(qX)
        lll = np.random.beta(2., 2.)
        rand_index = torch.randperm(qX[0].size()[0]).cuda()
        bbx1, bby1, bbx2, bby2 = rand_bbox(qX[0].size(), lll)
        ls = []
        for ii in range(b):
            label_a[ii] = labels[ii]
            label_b[ii] = labels[ii][rand_index]
            new_data[ii] = qX[ii]
            new_data[ii][:, :, bbx1:bbx2, bby1:bby2] = qX[ii][rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lll = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (qX[ii].size()[-1] * qX[ii].size()[-2]))  
            ls.append(lll)
        X = torch.cat([sX,new_data],1)
        X = X.reshape(N, C, H, W)
    X = X + data - data.detach()
    return X, label_a, label_b, torch.tensor(ls)

def cutmix(data, labels):
    label_a, label_b = torch.zeros_like(labels), torch.zeros_like(labels)
    new_data = torch.zeros_like(data)
    b = len(data)
    ls = []
    for ii in range(b):
        lll = np.random.beta(2., 2.)
        rand_index = torch.randperm(data[ii].size()[0]).cuda()
        label_a[ii] = labels[ii]
        label_b[ii] = labels[ii][rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(data[ii].size(), lll)
        new_data[ii] = data[ii]
        new_data[ii][:, :, bbx1:bbx2, bby1:bby2] = data[ii][rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lll = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data[ii].size()[-1] * data[ii].size()[-2]))  
        ls.append(lll)
    return new_data, label_a, label_b, torch.tensor(ls)


def augment_ops(img_shape):
    op_list =  [(shift_r,0.0,1.0),
                    (shift_g,0.0,1.0),
                    (shift_b,0.0,1.0),
                    (random_brightness,0.0,1.0),
                    (random_contrast,0.1,10.0),
                    (solarize,0.0,1.0),
                    (hflip, None, None),
                    (vflip, None, None),
                    (rotate,0.0,180.0),
                    (shift_x,0.0,0.5),
                    (shift_y,0.0,0.5),
                    (scale,0.1,10.0),
                    (self_mix, 0.1, 0.8),
                    (posterize, 0.0, 1.0),
                    (sample_pairing, 0.0,1.0),
                    (equalize, None, None),
                    (cutout_fixed_num_holes, 8, img_shape), # num_holes
                    (cutout_fixed_size,0.0,8.0,8.0,img_shape) #min_holes, max_holes,hole_size_divider
                    ]
    mag_mask = []
    for i, op in enumerate(op_list):
        if op[1] is not None:
            mag_mask.append(1.0)
        else:
            mag_mask.append(0.0)
    return op_list, mag_mask

class L2TT(nn.Module):
    def __init__(self, img_shape, mean, std, ops=augment_ops, max_op_len = 5, temp=1.0, opt=None):
        super(L2TT, self).__init__()
        self.L = max_op_len
        self.ops, self.mag_mask = ops(img_shape)
        self.mag_mask = torch.tensor(self.mag_mask).cuda()
        self.num_ops = len(self.ops)

        for s in range(self.L, self.L+1):
            setattr(self, 'probs{:d}'.format(s), nn.ParameterList([nn.Parameter(torch.zeros([self.num_ops]*(i+1))) for i in range(s)]))
            setattr(self, 'mag_params{:d}'.format(s), nn.ParameterList([nn.Parameter(torch.zeros([self.num_ops]*(i+1))) for i in range(s)]))
        
            
        self.temp = temp
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std',torch.tensor(std))

    def entropy(self):
        ent_arr = []
        for s in range(self.L, self.L+1):
            probs = getattr(self, 'probs{:d}'.format(s))
            jp = 1
            log_jp = 0
            for l in range(s):
                if l == 0:
                    jp = jp * F.softmax(probs[l],dim=-1)
                    log_jp = log_jp + F.log_softmax(probs[l],dim=-1)
                else:
                    jp = jp.unsqueeze(l) * F.softmax(probs[l],dim=-1)
                    log_jp = log_jp.unsqueeze(l) + F.log_softmax(probs[l],dim=-1)
            ent = (-jp * log_jp).sum()
            ent_arr.append(ent)
        ent_arr = torch.stack(ent_arr,0)
        entropy = ent_arr.mean()
        return entropy

    def avg_mag(self):
        mag_arr = []
        for s in range(self.L, self.L+1):
            mag_params = getattr(self, 'mag_params{:d}'.format(s))
            prob_params = getattr(self, 'probs{:d}'.format(s))
            avg_m  = torch.tensor(0.0).to(mag_params[0].device)
            all_p  = torch.tensor(1.0).to(mag_params[0].device)
            for l in range(s):
                m = torch.sigmoid(mag_params[l])
                p = F.softmax(prob_params[l],-1)
                all_p = all_p.unsqueeze(-1) * p
                mag_mask = self.mag_mask.view([1]*(m.dim()-1)+[self.num_ops])
                avg_m = avg_m.unsqueeze(-1) + m * mag_mask
            avg_m = (avg_m/s).view(-1).mean()
            # avg_m = ((avg_m/s).view(-1) * all_p.view(-1)).sum()
            mag_arr.append(avg_m)
        mag_arr = torch.stack(mag_arr,0)
        avg_mag = mag_arr.mean()

        
        return avg_mag
    def get_avg_magnitude(self):
        msg = 'Op(prob, mag), length 1 to {}\n'.format(self.L)
        mask_strs = ['S', 'Q', 'T']
 
        for l in range(self.L, self.L+1):
            set_sel = 'T'
            msg += '[L={}] '.format(l)
            probs = getattr(self, 'probs{:d}'.format(l))
            mag_params = getattr(self, 'mag_params{:d}'.format(l))
            ops = []
            max_probs = []
            mags = []
            
            vals = F.softmax(probs[0],-1)
            op = torch.argmax(vals,-1)
            max_probs.append(vals[op])
            mag = torch.sigmoid(mag_params[0][op])
            mags.append(mag)
            ops.append(op)
            for i in range(1,l): 
                vals = F.softmax(probs[i][(ops)],-1)
                op = torch.argmax(vals,-1)
                mag = torch.sigmoid(mag_params[i][(ops)])
                mags.append(mag[op].item())
                ops.append(op.item())
                max_probs.append(vals[op].item())
            msg += '-'.join(['{}({:.2f},{:.2f})'.format(self.ops[o][0].__name__,p,m) for o, p, m in zip(ops,max_probs,mags)])
            msg += '\n'
        return msg
    def unnormalize(self, x):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        ex_shape = [1]*len(x.shape[:-3])+[3,1,1]
        return x *std.view(*ex_shape) + mean.view(*ex_shape)

    def normalize(self, x):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        ex_shape = [1]*len(x.shape[:-3])+[3,1,1]
        return (x - mean.view(*ex_shape))/std.view(*ex_shape)

    def augment_batch_exhaustive(self, data, label, steps, set_sel, batch_size):
        op_index = []
        ent_arr = []
        mag_arr = []
        aug_probs = getattr(self, 'probs{:d}'.format(steps))
        mag_params = getattr(self, 'mag_params{:d}'.format(steps))
        for s in range(steps):
            if s == 0:
                probs = aug_probs[s]
                num_ops = len(probs)
                mags = torch.sigmoid(mag_params[s])

                ent_arr.append((-F.softmax(probs,-1)* F.log_softmax(probs,-1)).sum(-1))
                mag_arr.append((F.softmax(probs,-1)*mags*self.mag_mask).sum())
    
            else:
                probs = aug_probs[s][op_index]
                mags = torch.sigmoid(mag_params[s][op_index])
            
                ent_arr.append((-F.softmax(probs,-1)* F.log_softmax(probs,-1)).sum(-1))
                mag_arr.append((F.softmax(probs,-1)*mags*self.mag_mask).sum())
            
            sels = F.gumbel_softmax(probs,hard=True,tau=self.temp) # 1, num_ops  
            
            data = self.select_ops(data, label, sels, mags)

            op_index.append(torch.argmax(sels,-1))
        ent_arr = torch.stack(ent_arr,0)
        mag_arr = torch.stack(mag_arr,0)
        return data, ent_arr, mag_arr

    def select_ops(self, data, label, sels, mags):
        n_way = len(torch.unique(label))
        B, N, C, H, W = data.shape
        x = data.reshape(B*N, C, H, W)

        aug_data = [self.ops[i][0](x, mags[i], *self.ops[i][1:]) for i in range(self.num_ops)]
        aug_data = torch.stack(aug_data,dim=1) # BN, num_ops, C, H, W
        x = (aug_data * sels.view(1,self.num_ops,1,1,1)).sum(1)
        x = x.reshape(B, N, C, H, W)
        return x

    def _forward(self, support, query, labelS, labelQ):
        batch_size = len(support)
        Ns = support.shape[1]

        data = torch.cat((support,query),dim=1)
        label = torch.cat((labelS,labelQ),dim=1)
      
        data = self.unnormalize(data)

        support_num = support.shape[1]

        # steps = int(torch.randint(1,self.L+1,(1,))[0])
        steps = self.L

        new_data, ent, avg_mag = self.augment_batch_exhaustive(data, label, steps, None, batch_size)
      
        new_data = self.normalize(new_data)
       
        new_support = new_data[:,0:support_num]
        new_query = new_data[:,support_num:]
       
        return new_support, new_query
    
    def forward(self, support, query, labelS, labelQ, aug_num=4):
        new_support, new_query = [], []

        for i in range(aug_num):
            nsupp, nquery = self._forward(support, query, labelS, labelQ)
            new_support.append(nsupp)
            new_query.append(nquery)
        new_support = torch.stack(new_support,0)
        new_query = torch.stack(new_query,0)
        
        return new_support, new_query

