import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
from tqdm import tqdm

class MAML:
    def __init__(self, model, update_num, inner_lr, loss_func, first_order_approx):
        super(MAML, self).__init__()
        print('Using MAML')
        self.model = model
        self.update_num = update_num
        self.inner_lr = inner_lr
        self.loss_func = loss_func
        self.first_order_approx = first_order_approx
        self.min_val_loss = 1e9
        self.iter_count = 0
    
    def to_device(self,device):
        self.model.to(device)
        self.device = device

    def fast_adapt_task(self, support_x, support_y, query_x, query_y):
        img_shape = support_x.shape[-3:]
        support_x = support_x.reshape(-1,*img_shape)
        query_x = query_x.reshape(-1,*img_shape)
        support_y = support_y.view(-1)
        query_y = query_y.view(-1)
        fast_parameters = list(self.model.parameters()) #the first gradient calcuated in line 45 is based on original weight
        for weight in self.model.parameters():
            weight.fast = None
        self.model.zero_grad()
        for _ in range(self.update_num):
            out = self.model(support_x)
            loss = self.loss_func(out, support_y)
            grads = torch.autograd.grad(loss, fast_parameters, create_graph=(not self.first_order_approx))
            if self.first_order_approx:
                grads = [g.detach() for g in grads]
            fast_parameters = []
            for k, weight in enumerate(self.model.parameters()):
                #for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py 
                if weight.fast is None:
                    weight.fast = weight - self.inner_lr * grads[k] #create weight.fast 
                else:
                    weight.fast = weight.fast - self.inner_lr * grads[k] 
                fast_parameters.append(weight.fast) 
        out = self.model(query_x)
        return out

    def fast_adapt(self, support_x, support_y, query_x, query_y):
        task_num = len(support_x)
        outputs = []
        for b in range(task_num):
            out = self.fast_adapt_task(support_x[b], support_y[b], query_x[b], query_y[b])
            outputs.append(out)
        outputs = torch.stack(outputs,0)
        return outputs