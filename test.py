# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.autograd import Variable

from tqdm import tqdm

from models.protonet_embedding import ProtoNetEmbedding
from models.ResNet12_embedding import resnet12

from models.classification_heads import ClassificationHead, R2D2Head

from utils import pprint, set_gpu, Timer, count_accuracy, log
from models.maml import MAML

import random
import numpy as np
import os
import pdb
class Linear_fw(torch.nn.Linear): #used in MAML to forward input with fast weight 
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out
def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
        # network = torch.nn.DataParallel(network)
    elif options.network == 'ResNet':
        if options.dataset == 'miniImageNet':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5).cuda()
            # network = torch.nn.DataParallel(network)
        else:
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2).cuda()
            # network = torch.nn.DataParallel(network)
    else:
        print ("Cannot recognize the network type")
        assert(False)
    if len(options.gpu) > 1:
        print('nn.DataParallel')
        network = torch.nn.DataParallel(network)
    # Choose the classification head
    if options.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()    
    elif options.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif options.head == 'R2D2':
        cls_head = R2D2Head().cuda()
    elif options.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    else:
        print ("Cannot recognize the classification head type")
        assert(False)
        
    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if 'dda' in options.load:
        load_obj = torch.load(options.aug_model)
        mean, std = load_obj['meta_augger']['mean'], load_obj['meta_augger']['std']
        mean = mean.tolist()
        std = std.tolist()
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_test = MiniImageNet(phase='test', mean_pix=mean, std_pix=std) if 'dda' in options.load else MiniImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_test = CIFAR_FS(phase='test', mean_pix=mean, std_pix=std) if 'dda' in options.load else CIFAR_FS(phase='test')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_test, data_loader)


def self_mix(data):
    size = data.size()
    W = size[-1]
    H = size[-2]
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    cut_w = W//2
    cut_h = H//2

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    while True:
        bbxn = np.random.randint(0, W-(bbx2-bbx1))
        bbyn = np.random.randint(0, H-(bby2-bby1))

        if bbxn != bbx1 or bbyn != bby1:
            break
    if (bbx2 - bbx1) == (bby2 - bby1):
        k = random.sample([0, 1, 2, 3], 1)[0]
    else:
        k = 0
    data[:, :, bbx1:bbx2, bby1:bby2] = torch.rot90(data[:, :, bbxn:bbxn + (bbx2-bbx1), bbyn:bbyn + (bby2-bby1)], k, [2,3])
    #data[:, :, bbx1:bbx2, bby1:bby2] = data[:, :, bbxn:bbxn + (bbx2-bbx1), bbyn:bbyn + (bby2-bby1)]

    return data

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def build_grid(source_size,target_size):
    k = float(target_size)/float(source_size)
    direct = torch.linspace(-k,k,target_size).unsqueeze(0).repeat(target_size,1).unsqueeze(-1)
    full = torch.cat([direct,direct.transpose(1,0)],dim=2).unsqueeze(0)

    return full.cuda()

def random_crop_grid(x,grid):
    delta = x.size(2)-grid.size(1)
    grid = grid.repeat(x.size(0),1,1,1).cuda()
    #Add random shifts by x
    grid[:,:,:,0] = grid[:,:,:,0]+ torch.FloatTensor(x.size(0)).cuda().random_(0, delta).unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) /x.size(2)
    #Add random shifts by y
    grid[:,:,:,1] = grid[:,:,:,1]+ torch.FloatTensor(x.size(0)).cuda().random_(0, delta).unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) /x.size(2)

    return grid

def random_cropping(batch, t):
    #Building central crop of t pixel size
    grid_source = build_grid(batch.size(-1),t)
    #Make radom shift for each batch
    grid_shifted = random_crop_grid(batch,grid_source)
    #Sample using grid sample
    sampled_batch = F.grid_sample(batch, grid_shifted, mode='nearest')

    return sampled_batch

def shot_aug(data_support, labels_support, n_support, method, opt):
    size = data_support.shape
    if method == "fliplr":
        n_support = opt.s_du * n_support
        data_shot = flip(data_support, -1)
        data_support = torch.cat((data_support, data_shot), dim = 1)
        labels_support = torch.cat((labels_support, labels_support), dim = 1)
    elif method == "random_crop":
        n_support = opt.s_du * n_support
        data_shot = F.pad(data_support.view([-1] + list(data_support.shape[-3:])), (4,4,4,4))
        data_shot = random_cropping(data_shot, 32)
        data_support = torch.cat((data_support, data_shot.view([size[0], -1] + list(data_support.shape[-3:]))), dim = 1)
        labels_support = torch.cat((labels_support, labels_support), dim = 1)
    return data_support, labels_support, n_support

def main(opt):
    (dataset_test, data_loader) = get_dataset(opt)

    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.way,
        nKbase=0,
        nExemplars=opt.shot, # num training examples per novel category
        nTestNovel=opt.query * opt.way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=1,
        epoch_size=opt.episode, # num of batches per epoch
    )

    set_gpu(opt.gpu)
    
    # Define the models
    (embedding_net, cls_head) = get_model(opt)
    
    # Load saved model checkpoints
    saved_models = torch.load(opt.load)
    model_dict = saved_models['embedding']
    if 'module' in list(model_dict.keys())[0]:
        model_dict = {k[7:]:model_dict[k] for k in model_dict}
    embedding_net.load_state_dict(model_dict)
    embedding_net.eval()
    cls_head.load_state_dict(saved_models['head'])
    cls_head.eval()
    if opt.dataset == 'CIFAR_FS':
        img_shape = [32,32]
        data_mean = [x/255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
        data_std = [x/255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
    elif opt.dataset == 'miniImageNet':
        img_shape = [84,84]
   
    # Evaluate on test set
    test_accuracies = []
    if opt.verbose == 'show':
        loader_test = tqdm(dloader_test())
    else:
        loader_test = dloader_test()
    for i, batch in enumerate(loader_test, 1):
        data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]
        n_support = opt.way * opt.shot
        n_query = opt.way * opt.query

        for method in opt.shot_aug:
            data_support, labels_support, n_support = shot_aug(data_support, labels_support, n_support, method, opt)
        #augment tasks
        with torch.no_grad():
            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, n_support, -1)
        
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, n_query, -1)

        if opt.head == 'SVM':
            logits = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot, maxIter=3)
        else:
            logits = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot)

        acc = count_accuracy(logits.reshape(-1, opt.way), labels_query.reshape(-1))
        test_accuracies.append(acc.item())
        
        avg = np.mean(np.array(test_accuracies))
        std = np.std(np.array(test_accuracies))
        ci = std / np.sqrt(i + 1)
        if opt.verbose == 'show':
            if i % 50 == 0:
                print('Episode [{}/{}]:\t\t\tAccuracy: {:.2f} ± {:.2f} % ({:.2f} %)'\
                    .format(i, opt.episode, avg, ci, acc))
    return avg, ci


def maml_main(opt):
    class Flatten(torch.nn.Module):
        def __init__(self):
            super(Flatten, self).__init__()
            
        def forward(self, x):        
            return x.view(x.size(0), -1)
    (dataset_test, data_loader) = get_dataset(opt)

    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.way,
        nKbase=0,
        nExemplars=opt.shot, # num training examples per novel category
        nTestNovel=opt.query * opt.way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=1,
        epoch_size=opt.episode, # num of batches per epoch
    )

    set_gpu(opt.gpu)
    
    # Define the models
    if opt.network == 'ResNet':
        if opt.dataset == 'miniImageNet':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5, maml=True)
            fea_dim = 640 * 25
        else:
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2, maml=True)
            fea_dim = 640 * 4
    elif opt.network == 'ProtoNet':
        network = ProtoNetEmbedding(maml=True)
        if opt.dataset == 'miniImageNet':
            fea_dim = 1600
        else:
            fea_dim = 256
    else:
        print ("Cannot recognize the network type")
        assert(False)
    
    model = torch.nn.Sequential(network, Flatten(), Linear_fw(fea_dim, opt.way))

    # Load saved model checkpoints
    saved_models = torch.load(opt.load)
    model_dict = saved_models['model']
    if 'module' in list(model_dict.keys())[0]:
        model_dict = {k[7:]:model_dict[k] for k in model_dict}
    model.load_state_dict(model_dict)
    model.cuda()
    model.eval()
    maml_model = MAML(model, 10, 0.01, torch.nn.CrossEntropyLoss(), False)
    if opt.dataset == 'CIFAR_FS':
        img_shape = [32,32]
        data_mean = [x/255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
        data_std = [x/255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
    elif opt.dataset == 'miniImageNet':
        img_shape = [84,84]
    
    # Evaluate on test set
    test_accuracies = []
    if opt.verbose == 'show':
        loader_test = tqdm(dloader_test())
    else:
        loader_test = dloader_test()
    for i, batch in enumerate(loader_test, 1):
        data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]
        n_support = opt.way * opt.shot
        n_query = opt.way * opt.query

        #augment tasks
        logits = maml_model.fast_adapt(data_support, labels_support, data_query, labels_query)

        acc = count_accuracy(logits.reshape(-1, opt.way), labels_query.reshape(-1))
        test_accuracies.append(acc.item())
        
        avg = np.mean(np.array(test_accuracies))
        std = np.std(np.array(test_accuracies))
        ci = std / np.sqrt(i + 1)
        if opt.verbose == 'show':
            if i % 50 == 0:
                print('Episode [{}/{}]:\t\t\tAccuracy: {:.2f} ± {:.2f} % ({:.2f} %)'\
                    .format(i, opt.episode, avg, ci, acc))
    return avg, ci

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load', default='',
                            help='path of the checkpoint file')
    parser.add_argument('--aug_model', default='./experiments/exp_1/best_model.pth',
                            help='path of the augmentation model')
    parser.add_argument('--episode', type=int, default=1000,
                            help='number of episodes to test')
    parser.add_argument('--way', type=int, default=5,
                            help='number of classes in one test episode')
    parser.add_argument('--shot', type=int, default=1,
                            help='number of support examples per training class')
    parser.add_argument('--shot_aug', '-shotaug', default=[], nargs='+', type=str,
                            help='If use shot level data augmentation.')
    parser.add_argument('--s_du', type=int, default=1,
                            help='number of support examples augmented by shot')
    parser.add_argument('--query', type=int, default=15,
                            help='number of query examples per training class')
    parser.add_argument('--aug_num', type=int, default=0,
                            help='number of query examples per training class')
    parser.add_argument('--network', type=str, default='ResNet',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='R2D2',
                            help='choose which embedding network to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='CIFAR_FS',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--verbose', type=str, default='show',
                            help='show intermediate results')

    opt = parser.parse_args()
    main(opt)
