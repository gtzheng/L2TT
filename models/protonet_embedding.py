import torch.nn as nn
import math
import torch.nn.functional as F
import torch

class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out
            
class BatchNorm2d_fw(nn.BatchNorm2d): #used in MAML to forward input with fast weight 
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
            #batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, retain_activation=True, maml=False):
        super(ConvBlock, self).__init__()
        if maml:
            self.block = nn.Sequential(
                Conv2d_fw(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                BatchNorm2d_fw(out_channels)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        if retain_activation:
            self.block.add_module("ReLU", nn.ReLU(inplace=True))
        self.block.add_module("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        
    def forward(self, x):
        out = self.block(x)
        return out

class Conv4LayerNorm(nn.Module):
    def __init__(self, in_dim, x_dim=3, h_dim=64, z_dim=64, retain_last_activation=True):
        super(Conv4LayerNorm, self).__init__()
        in_shape1 = Conv4LayerNorm.calc_activation_shape(in_dim,(3,3),padding=(1,1))
        in_shape2 = Conv4LayerNorm.calc_activation_shape(in_shape1,(3,3),stride=(2,2),padding=(1,1))
        in_shape3 = Conv4LayerNorm.calc_activation_shape(in_shape2,(3,3),stride=(2,2),padding=(1,1))
        in_shape4 = Conv4LayerNorm.calc_activation_shape(in_shape3,(3,3),stride=(2,2),padding=(1,1))
        self.fea_dim = z_dim*(in_shape4[0]*in_shape4[1]//4)
        self.encoder = nn.Sequential(
            nn.Conv2d(x_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LayerNorm([h_dim,*in_shape1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LayerNorm([h_dim,*in_shape2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LayerNorm([h_dim,*in_shape3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(h_dim, z_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LayerNorm([z_dim,*in_shape4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @staticmethod
    def calc_activation_shape(dim,ksize,dilation=(1,1),stride=(1,1),padding=(0,0)):
        def shape_each_dim(i):
            odim_i = dim[i] + 2 * padding[i] - dilation[i] * (ksize[i] - 1) - 1
            return int((odim_i / stride[i]) + 1)
        return shape_each_dim(0),shape_each_dim(1)

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
# Embedding network used in Matching Networks (Vinyals et al., NIPS 2016), Meta-LSTM (Ravi & Larochelle, ICLR 2017),
# MAML (w/ h_dim=z_dim=32) (Finn et al., ICML 2017), Prototypical Networks (Snell et al. NIPS 2017).

class ProtoNetEmbedding(nn.Module):
    def __init__(self, x_dim=3, h_dim=64, z_dim=64, retain_last_activation=True, maml=False):
        super(ProtoNetEmbedding, self).__init__()
        self.encoder = nn.Sequential(
          ConvBlock(x_dim, h_dim, maml=maml),
          ConvBlock(h_dim, h_dim, maml=maml),
          ConvBlock(h_dim, h_dim, maml=maml),
          ConvBlock(h_dim, z_dim, retain_activation=retain_last_activation, maml=maml),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)