import math
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

from .performer_attention import SelfAttention


class DSBN1d(nn.Module):
    """
    Domain-specific Batch Normalization
    
    Args:
        dims (int): Dimension of the features.
        batchs (int): Domain number.
    """
    def __init__(self, dims, batchs, eps=1e-5, momentum=0.1,affine=True,
                 track_running_stats=True):
        super().__init__()
        self.batchs = batchs
        self.bns = nn.ModuleList([nn.BatchNorm1d(dims, eps, momentum, affine, track_running_stats) for _ in range(batchs)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_label):
        outs = torch.zeros_like(x)
        if domain_label==None:
            outs = self.bns[0](x)
        else:
            for i in range(self.batchs):
                indices = np.where(domain_label.cpu().numpy()==i)[0]
                if len(indices) > 1:
                    outs[indices] = self.bns[i](x[indices])
                elif len(indices) == 1:
                    outs[indices] = x[indices]
        return outs

class encoder_scRNA(nn.Module):
    """
    VAE of scRNA encoder
    
    Args:
        x_dims (int): Input dimension.
        z_dims (int): Latent dimension.
    """
    def __init__(self, x_dims=2000, z_dims=32):
        super().__init__()
        self.self = SelfAttention(dim=1, heads=1, dim_head=16)
        self.fc = nn.Linear(x_dims, 1024)
        self.fc_mu = nn.Linear(1024, z_dims)
        self.fc_var = nn.Linear(1024, z_dims)
        
        self.bn = nn.BatchNorm1d(1024)
        self.act = nn.ReLU()
        
        self.x_dims = x_dims

    def reparameterize(self, mu, var):
        z = Normal(mu, var.sqrt()).rsample()
        return z

    def forward(self, x):
        x = self.self(x.unsqueeze(-1))
        h = self.act(self.bn(self.fc(x.squeeze(-1))))
        mu = self.fc_mu(h)
        var = torch.exp(self.fc_var(h))
        z = self.reparameterize(mu, var)
        return z, mu, var
                         
class decoder_scRNA(nn.Module):
    """
    VAE of scRNA decoder
    
    Args:
        x_dims (int): Input dimension.
        z_dims (int): Latent dimension.
        batchs (int): Domain number.
    """
    def __init__(self, x_dims=2000, pooling_dims=16, batchs=20):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool1d(pooling_dims)
        self.fc = nn.Linear(pooling_dims, x_dims)
        self.bn = DSBN1d(dims=x_dims, batchs=batchs)
        self.act = nn.Sigmoid()

    def forward(self, z, domain_label=None):
        z = self.pooling(z)
        recon_x = self.fc(z)
        recon_x = self.act(self.bn(recon_x,domain_label))
        return recon_x
    
