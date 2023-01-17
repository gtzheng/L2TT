import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import shutil

from metadatas import *

from models.classification_heads import ClassificationHead, R2D2Head
from models.protonet_embedding import ProtoNetEmbedding, Conv4LayerNorm
from models.ResNet12_embedding import resnet12 

from tqdm import tqdm
from utils import set_gpu, Timer, count_accuracy, count_accuracy_mixup, check_dir, log
from dda.aug_functions import L2TT

from dda.aug_functions import augment_ops
from test import main as eval_model

x_entropy = torch.nn.CrossEntropyLoss()

def rand_bbox(size, lam=0.5):
    W = size[-1]
    H = size[-2]
    cut_rat = np.sqrt(lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    ## uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(data, labels):
    b, aug_num, N, C, H, W = data.shape
    label_a, label_b = torch.zeros_like(labels), torch.zeros_like(labels)
    new_data = torch.zeros_like(data)
    ls = []
    for ii in range(b):
        lll = np.random.beta(2., 2.)
        rand_index = torch.randperm(data[ii,0].size()[0]).cuda()
        label_a[ii] = labels[ii]
        label_b[ii] = labels[ii][rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(data[ii,0].size(), lll)
        new_data[ii] = data[ii]
        new_data[ii][:, :, :, bbx1:bbx2, bby1:bby2] = data[ii][:, rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lll = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data[ii,0].size()[-1] * data[ii,0].size()[-2]))  
        ls.append(lll)
    return new_data+data-data.detach(), label_a, label_b, torch.tensor(ls)


class Averager:
    def __init__(self, pool_size, init_val):
        self.N = pool_size
        self.arr = np.ones(self.N) * init_val
        self.count = 0
    def add(self, x):
        if self.count < self.N:
            self.arr[self.count] = x
            self.count += 1
        else:
            self.count = 0
            self.arr[self.count] = x
    def avg(self):
        return self.arr.mean()

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def eval_parser(load, shot, head, network, dataset, gpu,verbose='show'):
    opt = Namespace(
                    episode=10,
                    query=15,
                    way=5,
                    shot=shot,
                    shot_aug=[],
                    gpu=gpu,
                    network=network,
                    head=head,
                    dataset=dataset,
                    load=load,
                    verbose=verbose
                )

    return opt



def count_accuracy_mixup(logits, label_a, label_b, lam):
    correct = 0.
    for i, l in enumerate(lam):
        pred = torch.argmax(logits[i], dim=1).view(-1)
        la = label_a[i].view(-1)
        lb = label_b[i].view(-1)
        correct += pred.eq(la).float().sum() * l + (1 - l) * pred.eq(lb).float().sum()
    accuracy = correct/len(label_a.view(-1)) * 100
    return accuracy

def mixup_criterion(opt, pred, y_a, y_b, lam):
    n_q = opt.train_way * opt.train_query
    b = len(y_a)
    pred = pred.reshape(-1,opt.train_way)
    logit = F.log_softmax(pred, dim=-1)
    loss_a = F.nll_loss(logit, y_a.reshape(-1),reduction='none')
    loss_b = F.nll_loss(logit, y_b.reshape(-1),reduction='none')
    loss = loss_a.view(b,-1) * lam.view(b,1).cuda() + loss_b.view(b,-1) * (1 - lam).view(b,1).cuda()
    return loss


def query_cutmix(data_query, labels_query):
    B = len(query)
    label_a, label_b = torch.zeros_like(labels_query), torch.zeros_like(labels_query)
    new_data_q = torch.zeros_like(data_query)
    ls = []
    for ii in range(B):
        lll = np.random.beta(2., 2.)
        rand_index = torch.randperm(data_query[ii].size()[0]).to(data_query.device)
        label_a[ii] = labels_query[ii]
        label_b[ii] = labels_query[ii][rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(data_query[ii].size(), lll)
        new_data_q[ii] = data_query[ii]
        new_data_q[ii][:, :, bbx1:bbx2, bby1:bby2] = data_query[ii][rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lll = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data_query[ii].size()[-1] * data_query[ii].size()[-2]))  
        ls.append(lll)
    return new_data_q, label_a, label_b, ls

        
def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding()
    elif options.network == 'ResNet':
        if options.dataset == 'miniImageNet':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5)
        else:
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2)
    else:
        print ("Cannot recognize the network type")
        assert(False)

    
    # Choose the classification head
    if options.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet')
    elif options.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge')
    elif options.head == 'R2D2':
        cls_head = R2D2Head()
    elif options.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS')
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
   
    return (network, cls_head)


def get_datasets(name, phase, args):
    if name == 'miniImageNet':
        dataset = MiniImageNet(phase=phase, augment='null')  
    elif name == 'CIFAR_FS':
        dataset = CIFAR_FS(phase=phase, augment='null')
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
    print(dataset)

    return dataset


def meta_update(embedding_net, cls_head, meta_augger, batch, optimizer, opt):
    data_support, labels_support, data_query, labels_query = [x.cuda() for x in batch]
    indexS = torch.argsort(labels_support,1)
    indexQ = torch.argsort(labels_query,1)
    data_support = torch.gather(data_support, 1, indexS.view(*labels_support.shape,1,1,1).expand(*data_support.shape))
    data_query = torch.gather(data_query, 1, indexQ.view(*labels_query.shape,1,1,1).expand(*data_query.shape))
    labels_support = torch.gather(labels_support, 1, indexS)
    labels_query = torch.gather(labels_query, 1, indexQ)

    train_n_support = opt.train_way * opt.train_shot
    train_n_query = opt.train_way * opt.train_query 
    batch_num = len(data_support)
    acc_ref = torch.tensor(0.0)
        
    new_support, new_query = meta_augger(data_support, data_query, labels_support, labels_query)
    
    new_query = new_query.permute(1,0,2,3,4,5)
    new_query, labels_a, labels_b, lamda_arr = cutmix(new_query, labels_query) #cutmix
    new_query = new_query.permute(1,0,2,3,4,5)

    emb_support_aug = embedding_net(new_support.reshape(-1, *new_support.shape[-3:]))
    emb_query_aug = embedding_net(new_query.reshape(-1, *new_query.shape[-3:]))

    emb_support_aug = emb_support_aug.reshape(1,batch_num, train_n_support, -1).mean(0)
    emb_query_aug = emb_query_aug.reshape(1,batch_num, train_n_query, -1).mean(0)
    
    if opt.head == 'SVM':
        logit_query = cls_head(emb_query_aug, emb_support_aug, labels_support.reshape(batch_num,-1), opt.train_way, opt.train_shot, maxIter=3) # B, N, C
    else:
        logit_query = cls_head(emb_query_aug, emb_support_aug, labels_support.reshape(batch_num,-1), opt.train_way, opt.train_shot) # B, N, C
    
    loss_aug = mixup_criterion(opt, logit_query, labels_a, labels_b, lamda_arr)
    acc_aug = count_accuracy_mixup(logit_query, labels_a, labels_b, lamda_arr)

    entropy = meta_augger.entropy()
    mag_avg = meta_augger.avg_mag()
    loss = loss_aug.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, loss_aug.mean(), acc_aug, entropy, mag_avg

def main(opt):
    trainset = get_datasets(opt.dataset, 'train', opt)
    valset = get_datasets(opt.dataset, 'val', opt)
    epoch_s = opt.episodes_per_batch * opt.num_per_batch

    dloader_train = FewShotDataloader(trainset, kway=opt.train_way, kshot=opt.train_shot, kquery=opt.train_query,
                                    batch_size=opt.episodes_per_batch, num_workers=opt.num_workers, epoch_size=epoch_s, shuffle=True)
    dloader_val_ex = FewShotDataloader(valset, kway=opt.train_way, kshot=opt.train_shot, kquery=opt.train_query,
                                  batch_size=opt.episodes_per_batch, num_workers=opt.num_workers, epoch_size=epoch_s, shuffle=True)


    dloader_val = FewShotDataloader(valset, kway=opt.test_way, kshot=opt.val_shot, kquery=opt.val_query,
                                  batch_size=1, num_workers=opt.num_workers, epoch_size=opt.val_episode, shuffle=False, fixed=False)
    
    set_gpu(opt.gpu)
   
    exp_name = 'l2tt_{}_{}_{}_s{}_{}'.format(opt.head,opt.network,opt.dataset,opt.step,opt.tag)
    save_path = os.path.join(opt.save_path, exp_name)
    check_dir(save_path)
    log_file_path = os.path.join(save_path, "train_log.txt")
    
    log(log_file_path, str(vars(opt)))
   
    (embedding_net, cls_head) = get_model(opt)
    embedding_net.cuda()
    cls_head.cuda()

    
    
    device = list(embedding_net.parameters())[0].device
    support, _, _, _ = next(iter(dloader_train(0)))
    meta_augger = L2TT(support.shape[-2:],trainset.mean, trainset.std, augment_ops, max_op_len=opt.step, temp=opt.temp, opt=opt).to(device)
    meta_augger_dict = {'img_shape':support.shape[-2:], 'mean':trainset.mean, 'std':trainset.std, 
                         'ops':augment_ops, 'max_op_len':opt.step, 'temp':opt.temp}
    
    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},{'params': cls_head.parameters()},{'params': meta_augger.parameters()}], lr=0.1, momentum=0.9,weight_decay=5e-4)
    log(log_file_path,'Jointly optimization \n')
    lr_scheduler = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,opt.num_epoch,eta_min=1e-6,verbose=False)]
    if len(opt.gpu) > 1:
        print('nn.DataParallel')
        embedding_net = torch.nn.DataParallel(embedding_net)
    

    max_val_acc = 0.0
    max_train_acc = 0.0
    timer = Timer()
    
    train_ref_acc_epoch = []
    train_ref_loss_epoch = []
    train_model_loss_epoch = []
   
    dis_ref_epoch = []
    dis_aug_epoch = []
    dis_diff_epoch = []
    train_policy_loss_epoch = []
    train_aug_loss_epoch = []
    train_ent_epoch = []
    train_mag_epoch = []
    train_aug_acc_epoch = []
    dis_aug_policy_epoch = []
    val_loss_epoch = []
    val_acc_epoch = []
    epoch_arr = []
    min_dis_aug = 1e8
    tolerance_count = 0
  
    weight_averager = Averager(100, 100)
    for epoch in range(1, opt.num_epoch + 1):
    
        train_ref_accuracies = []
        train_ref_losses = []
        train_model_losses = []
        dis_ref_arr = []
        dis_aug_arr = []
        train_policy_losses = []
        train_aug_losses = []
        train_aug_accuracies = []
        train_ent_arr = []
        train_mag_arr = []
        dis_aug_policy = []


        count = 0
        embedding_net.train()
        cls_head.train()
        meta_augger.train()
        with tqdm(enumerate(zip(dloader_train(epoch),dloader_val_ex(epoch)), 1),leave=False, total=len(dloader_train)) as pbar:
            for i, (batch, batch_ex) in pbar:
                
                loss, loss_aug, acc_aug, entropy, mag_avg = meta_update(embedding_net, cls_head, meta_augger, batch, optimizer, opt)
                train_ent_arr.append(entropy.item())
                train_mag_arr.append(mag_avg.item())
                train_aug_losses.append(loss_aug.item())
                train_model_losses.append(loss.item())
                train_aug_accuracies.append(acc_aug.item())
                pbar.set_postfix({"AL": '{0:.2f}'.format(np.array(train_model_losses).mean() if len(train_model_losses)>0 else 0),
                            "Acc":'{0:.2f}'.format(np.array(train_aug_accuracies).mean() if len(train_aug_accuracies)>0 else 0),
                            'E':'{0:.2f}'.format(np.array(train_ent_arr).mean() if len(train_ent_arr)>0 else 0),
                            'M':'{0:.2f}'.format(np.array(train_mag_arr).mean() if len(train_mag_arr)>0 else 0),
                            'L':'{0:.2f}'.format(np.array(train_aug_losses).mean() if len(train_aug_losses)>0 else 0)
                            })
                
                
                
        train_model_loss_avg = np.mean(np.array(train_model_losses)) if len(train_model_losses) > 0 else 0
        train_aug_losses_avg = np.mean(np.array(train_aug_losses)) if len(train_aug_losses) > 0 else 0
        train_aug_accuracies_avg = np.mean(np.array(train_aug_accuracies)) if len(train_aug_accuracies) > 0 else 0

        train_ent_avg = np.mean(np.array(train_ent_arr)) if len(train_ent_arr) > 0 else 0
        train_mag_avg = np.mean(np.array(train_mag_arr)) if len(train_mag_arr) > 0 else 0
        [s.step() for s in lr_scheduler]

        # evaluation
        _, _ = [x.eval() for x in (embedding_net, cls_head)]
        meta_augger.eval()
        val_accuracies = []
        val_losses = []
        
        for i, batch in enumerate(dloader_val(epoch), 1):
            data_support, labels_support, data_query, labels_query = [x.cuda() for x in batch]
            test_n_support = opt.test_way * opt.val_shot
            test_n_query = opt.test_way * opt.val_query
          
            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, test_n_support, -1)
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)
            if opt.head == 'SVM':
                logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot, maxIter=3)[0]
            else:
                logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot)[0]
            loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())
            
        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))
   
        save_obj = {
                    'embedding': embedding_net.state_dict(), 
                    'head': cls_head.state_dict(),
                    'meta_augger': meta_augger.state_dict(),
                    'meta_augger_args': meta_augger_dict
                }
        msg = 'Epoch[{}] Loss {:.2f} Acc {:.2f} L {:.2f} E {:.3f} M {:.2f}\n'.format(epoch,
                                                train_model_loss_avg,
                                                train_aug_accuracies_avg,
                                                train_aug_losses_avg,
                                                train_ent_avg,
                                                train_mag_avg
                                               )
        msg += meta_augger.get_avg_magnitude()
     
        if getattr(meta_augger, 'step_prob', None) is not None:
            msg += ' '.join(['{}({:.2f})'.format(i+1,p.item()) for i,p in enumerate(F.softmax(meta_augger.step_prob,dim=-1))])
            msg += '\n'
        if getattr(meta_augger, 'mask_prob', None) is not None:
            mask_prob = F.softmax(meta_augger.mask_prob,-1)
            msg += 'Mask Probs: Supp {:.2f}, Query {:.2f}, All {:.2f}\n'.format(*mask_prob.detach().cpu().numpy())
     
        msg += 'val {:.4f}|{:.2f}±{:.2f}, scale {:.2f}'.format(val_loss_avg,val_acc_avg,val_acc_ci95, cls_head.scale.data[0])
        
        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save(save_obj, os.path.join(save_path, 'best_model.pth'))
            log(log_file_path, msg+' (best)')
        else:
            log(log_file_path, msg)


        if epoch % opt.save_epoch == 0:
            train_model_loss_epoch.append(train_model_loss_avg)
            
            train_aug_acc_epoch.append(train_aug_accuracies_avg)
            train_aug_loss_epoch.append(train_aug_losses_avg)
            val_loss_epoch.append(val_loss_avg)
            val_acc_epoch.append(val_acc_avg)

            train_ent_epoch.append(train_ent_avg)
            train_mag_epoch.append(train_mag_avg)
            epoch_arr.append(epoch)
            record_dict = {
                'train_model_loss' : train_model_loss_epoch,
                'train_aug_acc': train_aug_acc_epoch,
                'train_aug_loss': train_aug_loss_epoch,
                'val_acc': val_acc_epoch,
                'val_loss': val_loss_epoch,
                'epoch_arr': epoch_arr,
                'train_ent':train_ent_epoch,
                'train_mag':train_mag_epoch
            }
            torch.save(record_dict, os.path.join(save_path, 'records.pth'))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))
  
    
    print('-----------Evaluation-------------')
    load_path = os.path.join(save_path, 'best_model.pth')
    eval_opt = eval_parser(load_path, 1, opt.head, opt.network, opt.dataset, '', opt.gpu)
    avg_1shot, std_1shot = eval_model(eval_opt)

    eval_opt = eval_parser(load_path, 5, opt.head, opt.network, opt.dataset, '', opt.gpu)
    avg_5shots, std_5shots = eval_model(eval_opt)

    with open(os.path.join(save_path,'result.txt'),'w') as f:
        f.write('[1-shot] {:.4f}+-{:.4f}\n'.format(avg_1shot,std_1shot))
        f.write('[5-shot] {:.4f}+-{:.4f}\n'.format(avg_5shots,std_5shots))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=2,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=1,
                            help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=5,
                            help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=5,
                            help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=6,
                            help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=20,
                            help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15,
                            help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,
                            help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=5,
                            help='number of classes in one test (or validation) episode')
    parser.add_argument('--save-path', default='./l2tt_experiments')
    parser.add_argument('--gpu', default='2')
    parser.add_argument('--network', type=str, default='ProtoNet',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='ProtoNet',
                            help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='CIFAR_FS',
                            help='choose which classification head to use. miniImageNet, CIFAR_FS')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                            help='number of episodes per batch')
    parser.add_argument('--num-per-batch', type=int, default=10,
                            help='number of task batches per train epoch')
    parser.add_argument('--save_freq', type=int, default=1,
                            help='augmentation number at test time')
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')
    parser.add_argument('--temp', type=float, default=1.0,
                            help='sampling temperature')
    parser.add_argument('--step', type=int, default=5,
                            help='maximum number of image operations')
    parser.add_argument('--train_method', type=str, default='random',
                            help='random')
    parser.add_argument('--num-workers', type=int, default=4,
                            help='number of query examples per validation class')
    parser.add_argument('--tag', type=str, default='test',
                            help='additional information')
    parser.add_argument('--verbose', type=str, default='show',
                            help='show progress')
    opt = parser.parse_args()
    
    main(opt)
