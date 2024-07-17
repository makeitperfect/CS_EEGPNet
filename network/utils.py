import torch

import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function

EPSILON = 1e-8

class Conv2dWithConstraint(nn.Conv2d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class Conv1dWithConstraint(nn.Conv1d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

class FSFN2d(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(FSFN2d, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

        self.offline_mode = True
    
    def online(self):
        self.offline_mode = False

    def offline(self):
        self.offline_mode = True

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def forward(self, input):
        if self.offline_mode is True:
            batch_mean = input.mean(dim=(0, 2, 3))
            batch_var = input.var(dim=(0, 2, 3), unbiased=False)
            
            self.running_mean = batch_mean
            self.running_var = batch_var

        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        norm_input = (input - batch_mean[None, :, None, None]) / torch.sqrt(batch_var[None, :, None, None] + self.eps)

        if self.affine:
            return norm_input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        else:
            return norm_input

class CurrentLimitingF(Function):

    @staticmethod
    def forward(ctx, x, alpha = 0.3):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output* ctx.alpha
        return output, None

class CRG_Block(nn.Module):
    def __init__(self,input_channel,rate = 2,f1 = 15,f2 = 35,f3 = 55) -> None:
        super(CRG_Block,self).__init__()
        
        self.temporal_conv_1 = nn.Sequential(
            nn.Conv2d(input_channel,rate*input_channel,kernel_size=(1,f1),padding = (0,f1//2), groups=input_channel, bias = False),
            # nn.BatchNorm2d(rate*input_channel),
            # nn.Sigmoid()
        )
        
        self.temporal_conv_2 = nn.Sequential(
            nn.Conv2d(input_channel,rate*input_channel,kernel_size=(1,f2),padding = (0,f2//2),groups=input_channel,bias = False),
            # nn.BatchNorm2d(rate*input_channel),
            # nn.Sigmoid()
        )
        
        self.temporal_conv_3 = nn.Sequential(
            nn.Conv2d(input_channel,rate*input_channel,kernel_size=(1,f3),padding = (0,f3//2),groups=input_channel,bias = False),
            # nn.BatchNorm2d(rate*input_channel),
            # nn.Sigmoid()
        )
        
    def forward(self,feature):
        h1 = self.temporal_conv_1(feature)
        h2 = self.temporal_conv_2(feature)
        h3 = self.temporal_conv_3(feature)

        return torch.cat([h1,h2,h3],1)

def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """
    Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    # print(x,y)
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances

    elif matching_fn == 'cosine':
        
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        # print(x.pow(2).sum(dim=1, keepdim=True),x)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)
        # print(expanded_x.shape,expanded_y.shape)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return  1 - cosine_similarities 

    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))

