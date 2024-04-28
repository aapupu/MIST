import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

def KL_Div(mu, var):
    """Calculate KL Divergence between two normal distributions."""
    kl_loss = kl_divergence(Normal(mu, var.sqrt()), 
                            Normal(torch.zeros_like(mu),torch.ones_like(var))).sum(dim=1).mean()
    return 0.5 * kl_loss

def scRNA_recon(recon_x, x):
    """Calculate reconstruction loss for scRNA."""
    return F.binary_cross_entropy(recon_x, x) * x.size(-1)


def categorical_recon(recon_x, x):
    """Calculate reconstruction loss using cross-entropy."""
    return F.cross_entropy(recon_x.view(-1, recon_x.size(-1)), x.reshape(-1), ignore_index=0,reduction='sum') / x.size(0)


def scTCR_recon(recon_bv,recon_bj,recon_cdr3b,
                recon_av,recon_aj,recon_cdr3a,
                bv, bj, cdr3b, av, aj, cdr3a, beta=[1, 0.1, 1]):
    """Calculate reconstruction loss for TCR."""
    bv_recon_loss = beta[0] * categorical_recon(recon_bv, bv)
    bj_recon_loss = beta[1] * categorical_recon(recon_bj, bj)
    cdr3b_recon_loss = beta[2] *  categorical_recon(recon_cdr3b, cdr3b)
    av_recon_loss = beta[0] * categorical_recon(recon_av, av)
    aj_recon_loss = beta[1] * categorical_recon(recon_aj, aj)
    cdr3a_recon_loss = beta[2] * categorical_recon(recon_cdr3a, cdr3a)
    
    # Combine individual reconstruction losses with weighted sum
    recon_loss = torch.sum(torch.stack((bv_recon_loss, bj_recon_loss, cdr3b_recon_loss, 
                                        av_recon_loss, aj_recon_loss, cdr3a_recon_loss)))
    return recon_loss


def euclidean_dist(mu_R, mu_T):
    """Calculate euclidean distance between two vectors.""" 
    return torch.linalg.norm(mu_R - mu_T,dim=1).mean()

def symmKL(mu_R, var_R, mu_T, var_T):
    """Calculate symmetric KL Divergence between two probability distributions."""
    kl_divergence_R_to_T = kl_divergence(Normal(mu_R, var_R.sqrt()), 
                                        Normal(mu_T, var_T.sqrt()))
    kl_divergence_T_to_R = kl_divergence(Normal(mu_T, var_T.sqrt()),
                                         Normal(mu_R, var_R.sqrt()))
    symmetric_kl_divergence = kl_divergence_R_to_T.sum(dim=1).mean() + kl_divergence_T_to_R.sum(dim=1).mean()
    return 0.5 * symmetric_kl_divergence

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) 

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss
