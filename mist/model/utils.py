import numpy as np
import pandas as pd
import random
import os

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
    # current_dir = os.path.dirname(__file__)
    # project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    # imgt_vdj = pd.read_csv(os.path.join(project_root, 'doc', 'imgt_pip_vdj.csv')) 
    # imgt_vdj['Gene'] = imgt_vdj['0'].apply(lambda x: x.replace('DV', '/DV').replace('OR', '/OR') if ('DV' in x) or ('OR' in x) else x)
    # bv_dict = make_gene_dict(list(imgt_vdj['Gene'][imgt_vdj['Gene'].str.contains('TRBV')]))
    # bj_dict = make_gene_dict(list(imgt_vdj['Gene'][imgt_vdj['Gene'].str.contains('TRBJ')]))
    # av_dict = make_gene_dict(list(imgt_vdj['Gene'][imgt_vdj['Gene'].str.contains('TRAV')]))
    # aj_dict = make_gene_dict(list(imgt_vdj['Gene'][imgt_vdj['Gene'].str.contains('TRAJ')]))
    aa_dict = make_aa_dict()
    bv_dict = {
        'X': 0,'TRBV1': 1,'TRBV10-1': 2,'TRBV10-2': 3,'TRBV10-3': 4,'TRBV11-1': 5,'TRBV11-2': 6,'TRBV11-3': 7,'TRBV12-1': 8,'TRBV12-2': 9,
        'TRBV12-3': 10,'TRBV12-4': 11,'TRBV12-5': 12,'TRBV13': 13,'TRBV14': 14,'TRBV15': 15,'TRBV16': 16,'TRBV17': 17,'TRBV18': 18,'TRBV19': 19,
        'TRBV2': 20,'TRBV20-1': 21,'TRBV20/OR9-2': 22,'TRBV21-1': 23,'TRBV21/OR9-2': 24,'TRBV22-1': 25,'TRBV22/OR9-2': 26,'TRBV23-1': 27,'TRBV23/OR9-2': 28,
        'TRBV24-1': 29,'TRBV24/OR9-2': 30,'TRBV25-1': 31,'TRBV26': 32,'TRBV26/OR9-2': 33,'TRBV27': 34,'TRBV28': 35,'TRBV29-1': 36,'TRBV29/OR9-2': 37,'TRBV3-1': 38,
        'TRBV30': 39,'TRBV4-1': 40,'TRBV4-2': 41,'TRBV4-3': 42,'TRBV5-1': 43,'TRBV5-2': 44,'TRBV5-3': 45,'TRBV5-4': 46,'TRBV5-5': 47,'TRBV5-6': 48,'TRBV5-7': 49,
        'TRBV5-8': 50,'TRBV6-1': 51,'TRBV6-2': 52,'TRBV6-4': 53,'TRBV6-5': 54,'TRBV6-6': 55,'TRBV6-7': 56,'TRBV6-8': 57,'TRBV6-9': 58,'TRBV7-1': 59,'TRBV7-2': 60,
        'TRBV7-3': 61,'TRBV7-4': 62,'TRBV7-5': 63,'TRBV7-6': 64,'TRBV7-7': 65,'TRBV7-8': 66,'TRBV7-9': 67,'TRBV8-1': 68,'TRBV8-2': 69,'TRBV9': 70,'TRBVA': 71,
        'TRBVB': 72
    }
    bj_dict = {
        'X': 0,'TRBJ1-1': 1,'TRBJ1-2': 2,'TRBJ1-3': 3,'TRBJ1-4': 4,'TRBJ1-5': 5,'TRBJ1-6': 6,'TRBJ2-1': 7,'TRBJ2-2': 8,'TRBJ2-2P': 9,'TRBJ2-3': 10,
        'TRBJ2-4': 11,'TRBJ2-5': 12,'TRBJ2-6': 13,'TRBJ2-7': 14
    }
    av_dict = {
        'X': 0,'TRAV1-1': 1,'TRAV1-2': 2,'TRAV10': 3,'TRAV11': 4,'TRAV12-1': 5,'TRAV12-2': 6,'TRAV12-3': 7,'TRAV13-1': 8,'TRAV13-2': 9,
        'TRAV14/DV4': 10,'TRAV15': 11,'TRAV16': 12,'TRAV17': 13,'TRAV18': 14,'TRAV19': 15,'TRAV2': 16,'TRAV20': 17,'TRAV21': 18,
        'TRAV22': 19,'TRAV23/DV6': 20,'TRAV24': 21,'TRAV25': 22,'TRAV26-1': 23,'TRAV26-2': 24,'TRAV27': 25,'TRAV28': 26,
        'TRAV29/DV5': 27,'TRAV3': 28,'TRAV30': 29,'TRAV31': 30,'TRAV32': 31,'TRAV33': 32,'TRAV34': 33,'TRAV35': 34,
        'TRAV36/DV7': 35,'TRAV37': 36,'TRAV38-1': 37,'TRAV38-2/DV8': 38,'TRAV39': 39,'TRAV4': 40,'TRAV40': 41,
        'TRAV41': 42,'TRAV5': 43,'TRAV6': 44,'TRAV7': 45,'TRAV8-1': 46,'TRAV8-2': 47,'TRAV8-3': 48,
        'TRAV8-4': 49,'TRAV8-5': 50,'TRAV8-6': 51,'TRAV8-7': 52,'TRAV9-1': 53,'TRAV9-2': 54}
    aj_dict = {
        'X': 0,'TRAJ1': 1,'TRAJ10': 2,'TRAJ11': 3,'TRAJ12': 4,'TRAJ13': 5,'TRAJ14': 6,'TRAJ15': 7,'TRAJ16': 8,
        'TRAJ17': 9,'TRAJ18': 10,'TRAJ19': 11,'TRAJ2': 12,'TRAJ20': 13,'TRAJ21': 14,'TRAJ22': 15,'TRAJ23': 16,
        'TRAJ24': 17,'TRAJ25': 18,'TRAJ26': 19,'TRAJ27': 20,'TRAJ28': 21,'TRAJ29': 22,'TRAJ3': 23,'TRAJ30': 24,
        'TRAJ31': 25,'TRAJ32': 26,'TRAJ33': 27,'TRAJ34': 28,'TRAJ35': 29,'TRAJ36': 30,'TRAJ37': 31,'TRAJ38': 32,
        'TRAJ39': 33,'TRAJ4': 34,'TRAJ40': 35,'TRAJ41': 36,'TRAJ42': 37,'TRAJ43': 38,'TRAJ44': 39,'TRAJ45': 40,
        'TRAJ46': 41,'TRAJ47': 42,'TRAJ48': 43,'TRAJ49': 44,'TRAJ5': 45,'TRAJ50': 46,'TRAJ51': 47,'TRAJ52': 48,
        'TRAJ53': 49,'TRAJ54': 50,'TRAJ55': 51,'TRAJ56': 52,'TRAJ57': 53,'TRAJ58': 54,'TRAJ59': 55,'TRAJ6': 56,
        'TRAJ60': 57,'TRAJ61': 58,'TRAJ7': 59,'TRAJ8': 60,'TRAJ9': 61
        }
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
