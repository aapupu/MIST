import torch
import torch.nn.functional as F


def rna_corr_coef(recon_x, x, dim = 1, reduce_dims = (-1,)):
    recon_x_centered = recon_x - recon_x.mean(dim = dim, keepdim = True)
    x_centered = x - x.mean(dim = dim, keepdim = True)
    corr_coef = F.cosine_similarity(recon_x_centered, x_centered, dim = dim).mean(dim = reduce_dims)
    return corr_coef.item()

def recon_evaluate(recon_x, x,  pad_idx=0):
    x_indices = (x.long() != pad_idx)
    x_predictions = torch.argmax(recon_x, dim=-1)
    recon_acc = (x[x_indices] == x_predictions[x_indices]).float().mean().item()
    return recon_acc

def tcr_recon_evaluate(recon_bv,recon_bj,recon_cdr3b,
                recon_av,recon_aj,recon_cdr3a,
                bv, bj, cdr3b, av, aj, cdr3a):
    
    bv_acc = recon_evaluate(recon_bv, bv)
    bj_acc = recon_evaluate(recon_bj, bj)
    cdr3b_acc = recon_evaluate(recon_cdr3b, cdr3b)
    av_acc = recon_evaluate(recon_av, av)
    aj_acc = recon_evaluate(recon_aj, aj)
    cdr3a_acc = recon_evaluate(recon_cdr3a, cdr3a)
    return bv_acc, bj_acc, cdr3b_acc, av_acc, aj_acc, cdr3a_acc
