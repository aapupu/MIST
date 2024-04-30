import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.init as init

def make_gene_dict(gene):
    padding_dict = {'X': 0}
    gene.sort()
    gene_dict=dict(zip(gene,list(range(1,len(gene)+1))))
    padding_dict.update(gene_dict)
    return padding_dict

def make_aa_dict():
    amino_acid_dict = {
        'X': 0,
        'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5,
        'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
        'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
        'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20,
        "Z": 21,
    }
    return amino_acid_dict

def make_TCR_dict():
    imgt_vdj = pd.read_csv('docs/imgt_pip_vdj.csv')
    imgt_vdj['Gene'] = imgt_vdj['0'].apply(lambda x: x.replace('DV', '/DV').replace('OR', '/OR') if ('DV' in x) or ('OR' in x) else x)
    aa_dict = make_aa_dict()
    bv_dict = make_gene_dict(list(imgt_vdj['Gene'][imgt_vdj['Gene'].str.contains('TRBV')]))
    bj_dict = make_gene_dict(list(imgt_vdj['Gene'][imgt_vdj['Gene'].str.contains('TRBJ')]))
    av_dict = make_gene_dict(list(imgt_vdj['Gene'][imgt_vdj['Gene'].str.contains('TRAV')]))
    aj_dict = make_gene_dict(list(imgt_vdj['Gene'][imgt_vdj['Gene'].str.contains('TRAJ')]))
    return {'AA':aa_dict, 'TRBV':bv_dict, 'TRBJ':bj_dict,
            'TRAV':av_dict, 'TRAJ':aj_dict}

def gene_to_vec(genelist, gene_dict):
    return np.array([gene_dict[gene] for gene in genelist])

def aa_to_vec(aa_seq, aa_dict):
    vec = np.array([aa_dict[aa] for aa in aa_seq])
    return vec

def cdr3_to_vec(cdr3, aa_dict, max_len=30, end=False):
    cdr3 = cdr3.replace(u'\xa0', u'').upper()
    k = len(cdr3)
    if k > max_len:
        raise ValueError(f'cdr3 {cdr3} has length {len(cdr3)} > 30.')
    if end == True:
        cdr3_padding = cdr3 + "Z" + "X" * (max_len - k)
    else:
        cdr3_padding = cdr3 + "X" * (max_len - k)
    vec = aa_to_vec(cdr3_padding, aa_dict)
    return vec

def tcr_to_vec(adata,  aa_dict=None, 
               bv_dict=None, bj_dict=None, 
               av_dict=None, aj_dict=None, 
               max_len=30):
    bv = gene_to_vec(adata.obs['IR_VDJ_1_v_call'].tolist(), bv_dict)
    bj = gene_to_vec(adata.obs['IR_VDJ_1_j_call'].tolist(), bj_dict)
    cdr3b = np.array([cdr3_to_vec(cdr3, aa_dict, max_len) for cdr3 in adata.obs['IR_VDJ_1_junction_aa'].tolist()])
    
    av = gene_to_vec(adata.obs['IR_VJ_1_v_call'].tolist(), av_dict)
    aj = gene_to_vec(adata.obs['IR_VJ_1_j_call'].tolist(), aj_dict)
    cdr3a = np.array([cdr3_to_vec(cdr3, aa_dict, max_len) for cdr3 in adata.obs['IR_VJ_1_junction_aa'].tolist()])
    return bv, bj, cdr3b, av, aj, cdr3a

def convert_to_cdr3_sequence(cdr3_gen, aa_dict, temperature=1):
    scaled_logits = cdr3_gen / temperature
    probabilities = torch.softmax(scaled_logits, dim=2)
    cdr3_indices = torch.multinomial(probabilities.view(-1, cdr3_gen.size(2)),
                                     num_samples=1).view(cdr3_gen.size(0), cdr3_gen.size(1))
    cdr3_sequence = []
    for indices in cdr3_indices:
        sequence = ''.join([list(aa_dict.keys())[index] for index in indices])
        cdr3_sequence.append(sequence)
    return cdr3_sequence

def convert_to_gene(gene_gen, gene_dict, temperature=1):
    scaled_logits = gene_gen / temperature
    probabilities = torch.softmax(scaled_logits, dim=-1)
    gene_indices = torch.multinomial(probabilities, num_samples=1)
    gene = [list(gene_dict.keys())[index] for index in gene_indices]
    return gene

def convert_to_TCR(recon_bv, recon_bj, recon_cdr3b, 
                   recon_av, recon_aj, recon_cdr3a, 
                   aa_dict, bv_dict, bj_dict, 
                    av_dict, aj_dict, temperature=1):
    """
    Convert the generated TCR components from logits to actual sequences using temperature scaling.

    Args:
        recon_bv (torch.Tensor): Logits for the reconstructed TCR beta chain variable region.
        recon_bj (torch.Tensor): Logits for the reconstructed TCR beta chain joining region.
        recon_cdr3b (torch.Tensor): Logits for the reconstructed TCR beta chain CDR3 region.
        recon_av (torch.Tensor): Logits for the reconstructed TCR alpha chain variable region.
        recon_aj (torch.Tensor): Logits for the reconstructed TCR alpha chain joining region.
        recon_cdr3a (torch.Tensor): Logits for the reconstructed TCR alpha chain CDR3 region.
        aa_dict (dict): Dictionary mapping indices to amino acids.
        bv_dict (dict): Dictionary mapping indices to TCR beta chain variable region gene names.
        bj_dict (dict): Dictionary mapping indices to TCR beta chain joining region gene names.
        av_dict (dict): Dictionary mapping indices to TCR alpha chain variable region gene names.
        aj_dict (dict): Dictionary mapping indices to TCR alpha chain joining region gene names.
        temperature (float): Temperature for temperature scaling.

    Returns:
        tuple: Tuple containing the converted TCR components as sequences.
    """
    recon_bv = convert_to_gene(recon_bv.detach().cpu(), bv_dict, temperature)
    recon_bj = convert_to_gene(recon_bj.detach().cpu(), bj_dict, temperature)
    recon_cdr3b = convert_to_cdr3_sequence(recon_cdr3b.detach().cpu(),aa_dict,temperature)
    recon_av = convert_to_gene(recon_av.detach().cpu(), av_dict, temperature)
    recon_aj = convert_to_gene(recon_aj.detach().cpu(), aj_dict, temperature)
    recon_cdr3a = convert_to_cdr3_sequence(recon_cdr3a.detach().cpu(),aa_dict,temperature)
    return recon_bv,recon_bj,recon_cdr3b,recon_av,recon_aj,recon_cdr3a

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
            
            
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
