import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .scRNAlayer import encoder_scRNA, decoder_scRNA
from .scTCRlayer import encoder_scTCR, decoder_scTCR
from .utils import convert_to_TCR, kaiming_init, EarlyStopping
from .loss import KL_Div, scRNA_recon, scTCR_recon, euclidean_dist, symmKL, mmd_rbf
from .metrics import rna_corr_coef, tcr_recon_evaluate
from .dataset import scRNADataset, scTCRDataset, DataLoaderX


cdr3_conv_config = [
    {'conv_params': {'in_channels': 1, 'out_channels': 16, 'kernel_size': (4, 4), 'stride': (2, 2), 'padding': (0, 0)}},  #[14, 31]
    {'conv_params': {'in_channels': 16, 'out_channels': 32, 'kernel_size': (4, 3), 'stride': (2, 2), 'padding': (0, 0)}}, #[6, 15]
    {'conv_params': {'in_channels': 32, 'out_channels': 64, 'kernel_size': (3, 3), 'stride': (1, 2), 'padding': (0, 0)}},  #[4,7]
]


cdr3_conv_transpose_config = [
    {'conv_transpose_params': {'in_channels': 16, 'out_channels': 32, 'kernel_size': (3, 3), 'stride': (1, 1), 'padding': (1, 1)}},
    {'conv_transpose_params': {'in_channels': 32, 'out_channels': 64, 'kernel_size': (3, 2), 'stride': (1, 1), 'padding': (0, 0)}}
]


class VAE_scRNA(nn.Module):
    """
    VAE for scRNA-seq data.

    Args:
        x_dims (int): Input dimension.
        z_dims (int): Latent dimension.
        batchs (int): Number of domains.
    """
    def __init__(self, x_dims=2000, z_dims=16, batchs=20):
        super().__init__()
        self.encoder = encoder_scRNA(x_dims, z_dims)
        self.decoder = decoder_scRNA(x_dims, z_dims, batchs)
        self.x_dims = x_dims
        self.z_dims = z_dims
        self.batchs = batchs
        
    def _encode(self, dataloader, mode='latent',eval=True, device='cuda'):
        """
        Encode scRNA-seq data using the trained model.

        Args:
            dataloader (DataLoader): PyTorch DataLoader containing the input data.
            mode (str): Mode ('latent' or 'recon').
            eval (bool): Set to True for evaluation mode, False for training mode.
            device (str): Device to perform encoding ('cuda' or 'cpu').

        Returns:
            numpy.ndarray: Encoded data.
        """
        self.to(device)
        if eval:
            self.eval()
            print('eval mode')
        else:   
            self.train()
        if mode == 'latent':
            outs = np.zeros((dataloader.dataset.shape[0], self.z_dims))
            
            for x, _, index in dataloader:
                x = x.double().to(device)
                z,_,_ = self.encoder(x)
                outs[index] = z.detach().cpu().numpy()
        elif mode == 'recon':
            outs = np.zeros((dataloader.dataset.shape[0], self.x_dims))
            for x, domain_label, index in dataloader:
                x = x.double().to(device)
                z, _, _ = self.encoder(x)
                recon_x = self.decoder(z, domain_label).detach().cpu().numpy()
                outs[index] = recon_x
        return outs       
     
    def fit(self, train_dataloader,  valid_dataloader,
              lr=1e-4, weight_decay=1e-3,
              max_epoch=500,  device='cuda', 
              patience=10, outdir=None, verbose=False):
        """
        Train the VAE model.

        Args:
            train_dataloader (DataLoader): PyTorch DataLoader containing the training data.
            valid_dataloader (DataLoader): PyTorch DataLoader containing the validation data.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): L2 regularization strength.
            max_epoch (int): Maximum number of training epochs.
            device (str): Device to perform training ('cuda' or 'cpu').
            patience (int): Number of epochs with no improvement to wait before early stopping.
            outdir (str): Directory to save the trained model.
            verbose (bool): If True, print training progress.

        """
        self.to(device)
        self.apply(kaiming_init)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        early_stopping = EarlyStopping(patience=patience, verbose=verbose, path=outdir if outdir else os.path.join(os.getcwd(),'VAE_RNA.pt'))
    
        t = tqdm(range(max_epoch), desc="Epochs")
        for epoch in t:
            self.train()
            epoch_loss = {'recon_loss': 0.0, 'kl_loss': 0.0}
            pcc = []   
            for idx, (x, domain_label, _) in enumerate(train_dataloader):          
                x, domain_label = x.double().to(device), domain_label.to(device)
                z, mu, var = self.encoder(x)

                recon_x = self.decoder(z, domain_label)
                recon_loss = scRNA_recon(recon_x, x)
                kl_loss = KL_Div(mu, var)
                pcc.append(rna_corr_coef(recon_x, x))
                loss = {'recon_loss':recon_loss, 'kl_loss':kl_loss} 
                
                optimizer.zero_grad()
                sum(loss.values()).backward()
                optimizer.step()
                
                for key in loss.keys():
                    epoch_loss[key] += loss[key].item()      
            epoch_loss = {key:value/(idx+1) for key, value in epoch_loss.items()}
            info = ','.join(['{}={:.3f}'.format(key, value) for key, value in epoch_loss.items()])
            info += ',pcc={:.3f}'.format(torch.Tensor(pcc).mean().item())
            t.set_postfix_str(info)

            if valid_dataloader is None:
                valid_loss = sum(epoch_loss.values())
            else:
                valid_loss = self.evalute(valid_dataloader, device)
            early_stopping(valid_loss, self)
            if early_stopping.early_stop:
                print('EarlyStopping: run {} epoch'.format(epoch+1))
                break
    
    def evalute(self, valid_dataloader, device='cuda'):
        self.eval()
        epoch_loss = {'recon_loss': 0.0, 'kl_loss': 0.0}
        for idx, (x, domain_label, _) in enumerate(valid_dataloader):
            x, domain_label = x.double().to(device), domain_label.to(device)
            z, mu, var = self.encoder(x)
            recon_x = self.decoder(z, domain_label)
            
            recon_loss = scRNA_recon(recon_x, x)
            kl_loss = KL_Div(mu, var)
            loss = {'recon_loss':recon_loss, 'kl_loss':kl_loss} 
            
            for key in loss.keys():
                epoch_loss[key] += loss[key].item()
        epoch_loss = {key:value/(idx+1) for key, value in epoch_loss.items()}
        return sum(epoch_loss.values())
            
    def load_model(self, path):
        """
        Load pre-trained model parameters from a file.

        Args:
            path (str): The file path to load the model from.
        """
        state_dict1 = torch.load(path)
        state_dict2 = self.state_dict()
        for key in state_dict1:
            if key in state_dict2:
                state_dict2[key] = state_dict1[key]
        self.load_state_dict(state_dict2)

    def gene_attn_weight(self, adata, n_samples=64, device='cuda'):
        """
        Compute gene-gene attention weights.

        Args:
            adata (AnnData): Annotated Data object containing scRNA-seq data.
            n_samples (int): Batch size.
            device (str): Device to perform computation ('cuda' or 'cpu').

        Returns:
            pandas.DataFrame: DataFrame containing gene-gene attention weights.
        """
        self.to(device)
        self.eval()
        attn_weight_init = np.zeros((self.x_dims, self.x_dims))
        scdata = scRNADataset(adata)
        dataloader = DataLoaderX(scdata,  batch_size=n_samples, drop_last=False, shuffle=False, num_workers=8)
        for x, _, _ in dataloader:
            x = x.double().to(device)
            _,attn_weight = self.encoder.self(x, output_attentions=True)
            attn_weight = attn_weight.detach().cpu()
            attn_weight = attn_weight.mean((0,1)).numpy()
            attn_weight_init += attn_weight
        attn_weight_init /= len(dataloader)
        np.fill_diagonal(attn_weight_init, 0)
        return pd.DataFrame(attn_weight_init,index=adata.var.index.tolist(),columns=adata.var.index.tolist())
    
    def celltype_attn_weight(self, adata, Celltype, n_samples=64, device='cuda'):
        """
        Compute gene-celltype attention weights.

        Args:
            adata (AnnData): Annotated Data object containing scRNA-seq data.
            Celltype (str): Name of the celltype column in adata.obs.
            n_samples (int): Batch size.
            device (str): Device to perform computation ('cuda' or 'cpu').

        Returns:
            pandas.DataFrame: DataFrame containing celltype attention weights.
        """
        self.to(device)
        self.eval()
        outs = pd.DataFrame(index=adata.var.index.tolist(),columns=adata.obs[Celltype].cat.categories)
        for celltype_value in adata.obs[Celltype].cat.categories:
            adata_Subs = adata[adata.obs[Celltype].isin([celltype_value])]
            attn_weight_init = np.zeros((self.x_dims))
            scdata = scRNADataset(adata_Subs)
            dataloader = DataLoaderX(scdata,  batch_size=n_samples, drop_last=False, shuffle=False, num_workers=8)
            for x, _, _ in dataloader:
                x = x.double().to(device)
                _,attn_weight = self.encoder.self(x, output_attentions=True)
                attn_weight = attn_weight.detach().cpu().mean((0,1)).numpy()
                attn_weight_init += attn_weight.mean(0)
            attn_weight_init /= len(dataloader)
            outs[celltype_value] = attn_weight_init
        return outs

class VAE_scTCR(nn.Module):
    """
    VAE for scRNA-seq data.

    Args:
        z_dims (int): Latent dimension.
        aa_size (int): Size of the amino acid vocabulary.
        aa_dims (int): Dimensionality of amino acid embeddings.
        max_len (int): Maximum length of CDR3 sequences.
        bv_size (int): Size of the variable beta gene vocabulary.
        bj_size (int): Size of the joining beta gene vocabulary.
        av_size (int): Size of the variable alpha gene vocabulary.
        aj_size (int): Size of the joining alpha gene vocabulary.
        gene_dims (int): Dimensionality of gene embeddings.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, z_dims=128,
                aa_size=21, aa_dims=64, max_len=30, 
                bv_size=None, bj_size=None,
                av_size=None, aj_size=None, 
                gene_dims=48, drop_prob=0.1):
        super().__init__()
        self.TCRencoder = encoder_scTCR(z_dims, aa_size, aa_dims, max_len, 
                                        bv_size, bj_size,av_size, aj_size, 
                                        gene_dims, drop_prob,cdr3_conv_config)
        self.TCRdecoder = decoder_scTCR(z_dims, aa_size, aa_dims, max_len, 
                                        bv_size, bj_size, av_size, aj_size, 
                                        gene_dims, drop_prob,
                                        aa_embedding_weight=self.TCRencoder.aa_embedding.embedding.weight,
                                        bv_embedding_weight=self.TCRencoder.bv_encode.embedding.weight,
                                        bj_embedding_weight=self.TCRencoder.bj_encode.embedding.weight,
                                        av_embedding_weight=self.TCRencoder.av_encode.embedding.weight,
                                        aj_embedding_weight=self.TCRencoder.aj_encode.embedding.weight,
                                        conv_transpose_configs=cdr3_conv_transpose_config)
        self.z_dims = z_dims
        self.aa_size, self.aa_dims, self.max_len = aa_size, aa_dims, max_len
        self.bv_size, self.bj_size = bv_size, bj_size
        self.av_size, self.aj_size = av_size, aj_size
        
    def _encode(self, dataloader, mode='latent', eval=True, device='cuda',
                    TCR_dict=None, temperature=1):
        """
        Encode scTCR-seq data using the trained model.

        Args:
            dataloader (DataLoader): PyTorch DataLoader containing the input data.
            mode (str): Mode ('latent' or 'recon').
            eval (bool): Set to True for evaluation mode, False for training mode.
            device (str): Device to perform encoding ('cuda' or 'cpu').
            TCR_dict (dict): Dictionary containing a dictionary of mapping indices to TCR.
            temperature (float): Temperature for temperature scaling.

        Returns:
            Encoded data.
        """
        self.to(device)
        if eval:
            self.eval()
            print('eval mode')
        else:
            self.train()
        if mode == 'latent':
            outs = np.zeros((dataloader.dataset.shape[0], self.z_dims))
      
            for tcr, index in dataloader:
                tcr = tcr.long().to(device)
                bv = tcr[:,0].squeeze(-1)
                bj = tcr[:,1].squeeze(-1)
                cdr3b = tcr[:, 2:(2 + self.max_len)]
                av = tcr[:, 2 + self.max_len].squeeze(-1)
                aj = tcr[:, 3 + self.max_len].squeeze(-1)
                cdr3a = tcr[:, (4 + self.max_len):]   
                     
                z, _, _ = self.TCRencoder(bv, bj, cdr3b,
                                          av, aj, cdr3a)
                outs[index] = z.detach().cpu().numpy()
        elif mode == 'recon':
            outs = pd.DataFrame()
            
            for tcr, index in dataloader:
                tcr = tcr.long().to(device)
                bv = tcr[:,0].squeeze(-1)
                bj = tcr[:,1].squeeze(-1)
                cdr3b = tcr[:, 2:(2 + self.max_len)]
                av = tcr[:, 2 + self.max_len].squeeze(-1)
                aj = tcr[:, 3 + self.max_len].squeeze(-1)
                cdr3a = tcr[:, (4 + self.max_len):]   
                     
                z, _, _ = self.TCRencoder(bv, bj, cdr3b,
                                          av, aj, cdr3a)
                recon_bv, recon_bj, recon_cdr3b, recon_av, recon_aj, recon_cdr3a = self.TCRdecoder(z)
                recon_scTCR = convert_to_TCR(recon_bv, recon_bj, recon_cdr3b, 
                                            recon_av, recon_aj, recon_cdr3a, 
                                            TCR_dict['AA'], 
                                            TCR_dict['TRBV'], TCR_dict['TRBJ'], 
                                            TCR_dict['TRAV'], TCR_dict['TRAJ'],  
                                            temperature)
                
                batch_df = pd.DataFrame({
                                'bv': recon_scTCR[0],
                                'bj': recon_scTCR[1],
                                'cdr3b': recon_scTCR[2],
                                'av': recon_scTCR[3],
                                'aj': recon_scTCR[4],
                                'cdr3a': recon_scTCR[5]
                }, index=index.numpy())
                outs = pd.concat([outs, batch_df])
        return outs
     
    def fit(self, train_dataloader,  valid_dataloader,
              lr=1e-4, weight_decay=1e-3,
              max_epoch=100,  device='cuda', 
              patience=20, outdir=None, verbose=False):
        """
        Train the VAE model.

        Args:
            train_dataloader (DataLoader): PyTorch DataLoader containing the training data.
            valid_dataloader (DataLoader): PyTorch DataLoader containing the validation data.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): L2 regularization strength.
            max_epoch (int): Maximum number of training epochs.
            device (str): Device to perform training ('cuda' or 'cpu').
            patience (int): Number of epochs with no improvement to wait before early stopping.
            outdir (str, optional): Output directory to save the model. Default is None.
            verbose (bool): If True, print training progress.
        """
        self.to(device)
        self.apply(kaiming_init)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        early_stopping = EarlyStopping(patience=patience, verbose=verbose, path= outdir if outdir else os.path.join(os.getcwd(),'VAE_Multi.pt'))

        t = tqdm(range(max_epoch), desc="Epochs")
        for epoch in t: 
            self.train()
            epoch_loss = {'recon_loss': 0.0, 'kl_loss': 0.0}
            acc = []
            for idx, (tcr, _) in enumerate(train_dataloader):
                tcr = tcr.long().to(device)
                bv = tcr[:,0].squeeze(-1)
                bj = tcr[:,1].squeeze(-1)
                cdr3b = tcr[:, 2:(2 + self.max_len)]
                av = tcr[:, 2 + self.max_len].squeeze(-1)
                aj = tcr[:, 3 + self.max_len].squeeze(-1)
                cdr3a = tcr[:, (4 + self.max_len):]              
                z, mu, var = self.TCRencoder(bv, bj, cdr3b,
                                             av, aj, cdr3a)
                recon_bv, recon_bj, recon_cdr3b, recon_av, recon_aj, recon_cdr3a = self.TCRdecoder(z)
                recon_loss = scTCR_recon(recon_bv, recon_bj, recon_cdr3b, 
                                        recon_av, recon_aj, recon_cdr3a,
                                        bv, bj, cdr3b, av, aj, cdr3a)
                kl_loss = KL_Div(mu, var)
                acc.append(tcr_recon_evaluate(recon_bv, recon_bj, recon_cdr3b, 
                                               recon_av, recon_aj, recon_cdr3a,
                                               bv, bj, cdr3b, av, aj, cdr3a))
                loss = {'recon_loss': recon_loss, 'kl_loss': kl_loss}
                
                optimizer.zero_grad()
                sum(loss.values()).backward()
                optimizer.step()
                
                for key in loss.keys():
                    epoch_loss[key] += loss[key].item()
                    
            epoch_loss = {key:value/(idx+1) for key, value in epoch_loss.items()}
            info = ','.join(['{}={:.3f}'.format(key, value) for key, value in epoch_loss.items()])
            info += ',bv_acc={:.3f}'.format(np.mean([v[0] for v in acc]))
            info += ',bj_acc={:.3f}'.format(np.mean([v[1] for v in acc]))
            info += ',cdr3b_acc={:.3f}'.format(np.mean([v[2] for v in acc]))
            info += ',av_acc={:.3f}'.format(np.mean([v[3] for v in acc]))
            info += ',aj_acc={:.3f}'.format(np.mean([v[4] for v in acc]))
            info += ',cdr3a_acc={:.3f}'.format(np.mean([v[5] for v in acc]))
            t.set_postfix_str(info)

            if valid_dataloader is None:
                valid_loss = sum(epoch_loss.values())
            else:
                valid_loss = self.evalute(valid_dataloader, device)
            early_stopping(valid_loss, self)
            if early_stopping.early_stop:
                print('EarlyStopping: run {} epoch'.format(epoch+1))
                break
            
    def evalute(self, valid_dataloader, device='cuda'):
        self.eval()
        epoch_loss = {'recon_loss': 0.0, 'kl_loss': 0.0}
        for idx, (tcr, _)  in enumerate(valid_dataloader):
            tcr = tcr.long().to(device)
            bv = tcr[:,0].squeeze(-1)
            bj = tcr[:,1].squeeze(-1)
            cdr3b = tcr[:, 2:(2 + self.max_len)]
            av = tcr[:, 2 + self.max_len].squeeze(-1)
            aj = tcr[:, 3 + self.max_len].squeeze(-1)
            cdr3a = tcr[:, (4 + self.max_len):]             
            z, mu, var = self.TCRencoder(bv, bj, cdr3b, av, aj, cdr3a)
            recon_bv, recon_bj, recon_cdr3b, recon_av, recon_aj, recon_cdr3a = self.TCRdecoder(z)
            recon_loss = scTCR_recon(recon_bv, recon_bj, recon_cdr3b, 
                                    recon_av, recon_aj, recon_cdr3a,
                                    bv, bj, cdr3b, av, aj, cdr3a)
            kl_loss = KL_Div(mu, var)
            loss = {'recon_loss':recon_loss, 'kl_loss':kl_loss} 
            for key in loss.keys():
                epoch_loss[key] += loss[key].item()
        epoch_loss = {key:value/(idx+1) for key, value in epoch_loss.items()}
        return sum(epoch_loss.values())
    
    def aa_embed(self, tensor, device):
        """
        Embed amino acid.

        Args:
            tensor (Tensor): Input tensor of amino acid index.
            device (str): Device to perform embedding ('cuda' or 'cpu').

        Returns:
            Embedded amino acid vector.
        """
        self.to(device)
        self.eval()
        tensor = tensor.to(device)
        outs = self.TCRencoder.aa_embedding.embedding(tensor)
        outs = outs.detach().squeeze(0).cpu().numpy()
        return outs
        
    def aa_attn_weight(self, adata, TCR_dict, batch_size, device='cuda'):
        """
        Compute attention weights for amino acid sequences.

        Args:
            adata (AnnData): Annotated Data object containing scTCR-seq data.
            TCR_dict (dict): Dictionary containing information about TCR vocabulary.
            batch_size (int): Batch size.
            device (str): Device to perform computation ('cuda' or 'cpu').

        Returns:
            cdr3b_df (pandas.DataFrame): DataFrame containing attention weights for CDR3b.
            cdr3a_df (pandas.DataFrame): DataFrame containing attention weights for CDR3a.
        """
        self.to(device)
        self.eval()
        scdata = scTCRDataset(adata,TCR_dict=TCR_dict)
        dataloader = DataLoaderX(scdata,  batch_size=batch_size, drop_last=False, shuffle=False, num_workers=8)
        cdr3b_attn_mean = np.zeros((self.max_len, self.max_len))
        cdr3a_attn_mean = np.zeros((self.max_len, self.max_len))
        for idx, (tcr, _) in enumerate(dataloader):
            tcr = tcr.long().to(device)
            cdr3b = tcr[:, 2:(2 + self.max_len)]
            cdr3a = tcr[:, (4 + self.max_len):]
            attn_mask_b = (cdr3b == 0).clone().detach()
            attn_mask_a = (cdr3a == 0).clone().detach()
            
            cdr3b, cdr3a = self.TCRencoder.aa_embedding(cdr3b), self.TCRencoder.aa_embedding(cdr3a)
            _,attn_weight_b = self.TCRencoder.cdr3b_encode.self(cdr3b,cdr3b,cdr3b, 
                                                                attn_mask=None, key_padding_mask=attn_mask_b)
            _,attn_weight_a = self.TCRencoder.cdr3a_encode.self(cdr3a,cdr3a,cdr3a, 
                                                                attn_mask=None, key_padding_mask=attn_mask_a)
        
            attn_weight_b = attn_weight_b.detach().cpu().numpy().mean(0)
            attn_weight_a = attn_weight_a.detach().cpu().numpy().mean(0)
            cdr3b_attn_mean += attn_weight_b
            cdr3a_attn_mean += attn_weight_a
        np.fill_diagonal(cdr3b_attn_mean, 0)
        np.fill_diagonal(cdr3a_attn_mean, 0)
        cdr3b_df = pd.DataFrame(cdr3b_attn_mean/ (idx+1), index=list(range(1, self.max_len+1)), columns=list(range(1, self.max_len+1)))
        cdr3a_df = pd.DataFrame(cdr3a_attn_mean/ (idx+1), index=list(range(1, self.max_len+1)), columns=list(range(1, self.max_len+1)))
        return cdr3b_df, cdr3a_df
    
    # def tcr_attn_weight(self, adata, Type, TCR_dict, batch_size, device='cuda'):
    #     self.to(device)
    #     self.eval()
    #     cdr3b_df = pd.DataFrame(index=list(range(1, self.max_len+1)), columns=adata.obs[Type].cat.categories)
    #     cdr3a_df = pd.DataFrame(index=list(range(1, self.max_len+1)), columns=adata.obs[Type].cat.categories)
    #     for type in adata.obs[Type].cat.categories:
    #         adata_sub = adata[adata.obs[Type].isin([type])]
    #         scdata = scTCRDataset(adata_sub,TCR_dict=TCR_dict)
    #         dataloader = DataLoaderX(scdata,  batch_size=batch_size, drop_last=False, shuffle=False, num_workers=8)
    #         cdr3b_attn_mean = np.zeros((self.max_len, self.max_len))
    #         cdr3a_attn_mean = np.zeros((self.max_len, self.max_len))
    #         for idx, (tcr, _) in enumerate(dataloader):
    #             tcr = tcr.long().to(device)
    #             cdr3b = tcr[:, 2:(2 + self.max_len)]
    #             cdr3a = tcr[:, (4 + self.max_len):]
    #             attn_mask_b = (cdr3b == 0).clone().detach()
    #             attn_mask_a = (cdr3a == 0).clone().detach()
                
    #             cdr3b, cdr3a = self.TCRencoder.aa_embedding(cdr3b), self.TCRencoder.aa_embedding(cdr3a)
    #             _,attn_weight_b = self.TCRencoder.cdr3b_encode.self(cdr3b,cdr3b,cdr3b, 
    #                                                                 attn_mask=None, key_padding_mask=attn_mask_b)
    #             _,attn_weight_a = self.TCRencoder.cdr3a_encode.self(cdr3a,cdr3a,cdr3a, 
    #                                                                 attn_mask=None, key_padding_mask=attn_mask_a)
            
    #             attn_weight_b = attn_weight_b.detach().cpu().numpy().mean(0)
    #             attn_weight_a = attn_weight_a.detach().cpu().numpy().mean(0)
    #             cdr3b_attn_mean += attn_weight_b
    #             cdr3a_attn_mean += attn_weight_a
    #         cdr3b_df[type] = cdr3b_attn_mean.mean(0) / (idx+1)
    #         cdr3a_df[type] = cdr3a_attn_mean.mean(0) / (idx+1)
    #     return cdr3b_df, cdr3a_df
    
    def load_model(self, path):
        """
        Load pre-trained model parameters from a file.

        Args:
            path (str): The file path to load the model from.
        """
        state_dict1 = torch.load(path)
        state_dict2 = self.state_dict()
        for key in state_dict1:
            if key in state_dict2:
                state_dict2[key] = state_dict1[key]
        self.load_state_dict(state_dict2)
        
class VAE_Multi(nn.Module):
    """
    VAE for scRNA-seq and scTCR-seq data.

    Args:
        x_dims (int): Input dimension.
        z_dims (int): Latent dimension.
        pooling_dims (int): Dimensionality of pooling layer.
        batchs (int): Number of domains.
        aa_size (int): Size of the amino acid vocabulary.
        aa_dims (int): Dimensionality of amino acid embeddings.
        max_len (int): Maximum length of CDR3 sequences.
        bv_size (int): Size of the variable beta gene vocabulary.
        bj_size (int): Size of the joining beta gene vocabulary.
        av_size (int): Size of the variable alpha gene vocabulary.
        aj_size (int): Size of the joining alpha gene vocabulary.
        gene_dims (int): Dimensionality of gene embeddings.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, x_dims=2000, pooling_dims=16,
                 z_dims=128, batchs=20,
                aa_size=21, aa_dims=64, max_len=30, 
                bv_size=None, bj_size=None,
                av_size=None, aj_size=None, 
                gene_dims=48, drop_prob=0.1,
                weights=None):
        super().__init__()
        self.RNAencoder = encoder_scRNA(x_dims, z_dims)
        self.RNAdecoder = decoder_scRNA(x_dims, pooling_dims, batchs)
        
        self.TCRencoder = encoder_scTCR(z_dims, aa_size, aa_dims, max_len, 
                                        bv_size, bj_size,av_size, aj_size, 
                                        gene_dims, drop_prob,cdr3_conv_config)
        self.TCRdecoder = decoder_scTCR(z_dims, aa_size, aa_dims, max_len, 
                                        bv_size, bj_size, av_size, aj_size, 
                                        gene_dims, drop_prob,
                                        aa_embedding_weight=self.TCRencoder.aa_embedding.embedding.weight,
                                        bv_embedding_weight=self.TCRencoder.bv_encode.embedding.weight,
                                        bj_embedding_weight=self.TCRencoder.bj_encode.embedding.weight,
                                        av_embedding_weight=self.TCRencoder.av_encode.embedding.weight,
                                        aj_embedding_weight=self.TCRencoder.aj_encode.embedding.weight,
                                        conv_transpose_configs=cdr3_conv_transpose_config)
        
        if weights is not None:
            self.weights_mu = nn.Parameter(torch.ones(2))
            self.weights_var = nn.Parameter(torch.ones(2))
        else:
            self.weights_mu = weights
            self.weights_var = weights
            
        self.x_dims = x_dims
        self.z_dims = z_dims
        self.batchs = batchs

        self.aa_size,self.aa_dims, self.max_len = aa_size, aa_dims, max_len
        self.bv_size, self.bj_size = bv_size, bj_size
        self.av_size, self.aj_size = av_size, aj_size
        
    def _encodeTCR(self, dataloader, mode='latent', eval=True, device='cuda',
                   TCR_dict=None, temperature=1):
        self.to(device)
        if eval:
            self.eval()
            print('eval mode')
        else:
            self.train()
        if mode == 'latent':
            outs = np.zeros((dataloader.dataset.shape[0], self.z_dims))
            
            for tcr, index in dataloader:
                tcr = tcr.long().to(device)
                bv, bj, cdr3b = tcr[:,0].squeeze(-1), tcr[:,1].squeeze(-1),tcr[:, 2:(2 + self.max_len)]
                av, aj,cdr3a = tcr[:, 2 + self.max_len].squeeze(-1), tcr[:, 3 + self.max_len].squeeze(-1), tcr[:, (4 + self.max_len):]    
                zT,_, _ = self.TCRencoder(bv, bj, cdr3b, av, aj, cdr3a)
                outs[index] = zT.detach().cpu().numpy()
                
        elif mode == 'recon':
            outs = pd.DataFrame()
            for tcr, index in dataloader:
                tcr = tcr.long().to(device)
                bv, bj, cdr3b = tcr[:,0].squeeze(-1), tcr[:,1].squeeze(-1),tcr[:, 2:(2 + self.max_len)]
                av, aj,cdr3a = tcr[:, 2 + self.max_len].squeeze(-1), tcr[:, 3 + self.max_len].squeeze(-1), tcr[:, (4 + self.max_len):]    
                zT,_, _ = self.TCRencoder(bv, bj, cdr3b, av, aj, cdr3a)
                recon_bv, recon_bj, recon_cdr3b, recon_av, recon_aj, recon_cdr3a = self.TCRdecoder(zT)
                recon_scTCR = convert_to_TCR(recon_bv, recon_bj, recon_cdr3b, 
                                            recon_av, recon_aj, recon_cdr3a, 
                                            TCR_dict['AA'], 
                                            TCR_dict['TRBV'], TCR_dict['TRBJ'], 
                                            TCR_dict['TRAV'], TCR_dict['TRAJ'], 
                                            temperature)
                
                batch_df = pd.DataFrame({
                                'bv': recon_scTCR[0],
                                'bj': recon_scTCR[1],
                                'cdr3b': recon_scTCR[2],
                                'av': recon_scTCR[3],
                                'aj': recon_scTCR[4],
                                'cdr3a': recon_scTCR[5]
                }, index=index.numpy())
                outs = pd.concat([outs, batch_df])
        return outs
        
    def _encodeRNA(self, dataloader, mode='latent', eval=True, device='cuda'):
        self.to(device)
        if eval:
            self.eval()
            print('eval mode')
        else:
            self.train()
        if mode == 'latent':
            outs = np.zeros((dataloader.dataset.shape[0], self.z_dims))
            
            for x, _, index in dataloader:
                x = x.double().to(device)
                zR,_,_ = self.RNAencoder(x)
                outs[index] = zR.detach().cpu().numpy()
                
        elif mode == 'recon':
            outs = np.zeros((dataloader.dataset.shape[0], self.x_dims))
            for x, domain_label, index in dataloader:
                x = x.double().to(device)
                zR, _, _ = self.RNAencoder(x) 
                recon_scRNA = self.RNAdecoder(zR,domain_label=domain_label).detach().cpu().numpy()
                outs[index] = recon_scRNA
        return outs       
        
    def _encodeMulti(self, dataloader, mode='latent', eval=True, device='cuda',
                    TCR_dict=None, temperature=1):
        """
        Encode scRNA-seq and scTCR-seq data using the trained model.

        Args:
            dataloader (DataLoader): PyTorch DataLoader containing the input data.
            mode (str): Mode ('latent' or 'recon').
            eval (bool): Set to True for evaluation mode, False for training mode.
            device (str): Device to perform encoding ('cuda' or 'cpu').
            TCR_dict (dict): Dictionary contain dictionary of mapping indices to tcr.
            temperature (float): Temperature for temperature scaling.

        Returns:
            Encoded data.
        """
        self.to(device)
        if eval:
            self.eval()
            print('eval mode')
        else:
            self.train()
        if mode == 'latent':
            outs = np.zeros((dataloader.dataset.shape[0], self.z_dims))
    
            for rna, tcr,_, index in dataloader:
                rna = rna.double().to(device)
                tcr = tcr.long().to(device)
                bv = tcr[:,0].squeeze(-1)
                bj = tcr[:,1].squeeze(-1)
                cdr3b = tcr[:, 2:(2 + self.max_len)]
                av = tcr[:, 2 + self.max_len].squeeze(-1)
                aj = tcr[:, 3 + self.max_len].squeeze(-1)
                cdr3a = tcr[:, (4 + self.max_len):]   
                   
                _, mu_R, var_R = self.RNAencoder(rna)
                _, mu_T, var_T = self.TCRencoder(bv, bj, cdr3b,
                                                  av, aj, cdr3a)
                z, _, _ = self.reparameterize(mu_R, mu_T, var_R, var_T,
                                              weights_mu=self.weights_mu, weights_var=self.weights_var)
                outs[index] = z.detach().cpu().numpy()
        elif mode == 'recon':
            outs_scRNA = np.zeros((dataloader.dataset.shape[0], self.x_dims))
            outs_scTCR = pd.DataFrame()
            
            for rna, domain_label, index, bv, bj, cdr3b, av, aj, cdr3a in dataloader:
                rna = rna.double().to(device)
                domain_label = domain_label().to(device)
                tcr = tcr.long().to(device)
                bv = tcr[:,0].squeeze(-1)
                bj = tcr[:,1].squeeze(-1)
                cdr3b = tcr[:, 2:(2 + self.max_len)]
                av = tcr[:, 2 + self.max_len].squeeze(-1)
                aj = tcr[:, 3 + self.max_len].squeeze(-1)
                cdr3a = tcr[:, (4 + self.max_len):]        
                 
                _, mu_R, var_R = self.RNAencoder(rna)
                _, mu_T, var_T = self.TCRencoder(bv, bj, cdr3b,
                                                  av, aj, cdr3a)
                z, _, _ = self.reparameterize(mu_R, mu_T, var_R, var_T)
                
                recon_rna = self.RNAdecoder(z, domain_label).detach().cpu().numpy()
                outs_scRNA[index] = recon_rna
                
                recon_bv, recon_bj, recon_cdr3b, recon_av, recon_aj, recon_cdr3a = self.TCRdecoder(z)
                recon_scTCR = convert_to_TCR(recon_bv, recon_bj, recon_cdr3b, 
                                            recon_av, recon_aj, recon_cdr3a, 
                                            TCR_dict['AA'], 
                                            TCR_dict['TRBV'], TCR_dict['TRBJ'], 
                                            TCR_dict['TRAV'], TCR_dict['TRAJ'],  
                                            temperature)
                
                batch_df = pd.DataFrame({
                                'bv': recon_scTCR[0],
                                'bj': recon_scTCR[1],
                                'cdr3b': recon_scTCR[2],
                                'av': recon_scTCR[3],
                                'aj': recon_scTCR[4],
                                'cdr3a': recon_scTCR[5]
                }, index=index.numpy())
                outs_scTCR = pd.concat([outs_scTCR, batch_df])
            outs = {'scRNA':outs_scRNA, 'scTCR':outs_scTCR}    
        return outs
     
    def fit(self, train_dataloader,  valid_dataloader,
              lr=1e-4, weight_decay=1e-3,
              max_epoch=400,  device='cuda', 
              penalty='mmd_rbf',
              patience=40, warmup=40 ,
              outdir=None, verbose=False,
              pretrain=None):
        """
        Train the VAE model.

        Args:
            train_dataloader (DataLoader): PyTorch DataLoader containing the training data.
            valid_dataloader (DataLoader): PyTorch DataLoader containing the validation data.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): L2 regularization strength.
            max_epoch (int): Maximum number of training epochs.
            device (str): Device to perform training ('cuda' or 'cpu').
            penalty (str): Way to penalty latents ('euclidean_distance', 'symmKL' or 'euclidean_dist').
            patience (int): Number of epochs with no improvement to wait before early stopping.
            verbose (bool): If True, print training progress.
        """
        self.apply(kaiming_init)
        if pretrain is not None: self.load_modelpretrain()
        self.to(device)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        early_stopping = EarlyStopping(patience=patience, verbose=verbose, path= outdir if outdir else os.path.join(os.getcwd(),'VAE_Multi.pt'))

        t = tqdm(range(max_epoch), desc="Epochs")
        for epoch in t: 
            self.train()
            epoch_loss = {'scRNA_recon_loss': 0.0, 'scTCR_recon_loss': 0.0, 
                          'kl_loss': 0.0,'penalty_loss':0.0}
            acc, pcc = [], []
            for idx, (rna, tcr, domain_label, _) in enumerate(train_dataloader):
                rna = rna.double().to(device)
                tcr = tcr.long().to(device)
                bv = tcr[:,0].squeeze(-1)
                bj = tcr[:,1].squeeze(-1)
                cdr3b = tcr[:, 2:(2 + self.max_len)]
                av = tcr[:, 2 + self.max_len].squeeze(-1)
                aj = tcr[:, 3 + self.max_len].squeeze(-1)
                cdr3a = tcr[:, (4 + self.max_len):]      
                                                        
                zR, mu_R, var_R = self.RNAencoder(rna)
                zT, mu_T, var_T = self.TCRencoder(bv, bj, cdr3b,
                                                  av, aj, cdr3a)
                z, mu, var = self.reparameterize(mu_R, mu_T, var_R, var_T,
                                                  weights_mu=self.weights_mu, weights_var=self.weights_var)
                
                recon_rna = self.RNAdecoder(z, domain_label)
                recon_bv, recon_bj, recon_cdr3b, recon_av, recon_aj, recon_cdr3a = self.TCRdecoder(z)
                
                scRNA_recon_loss = scRNA_recon(recon_rna, rna)
                scTCR_recon_loss = scTCR_recon(recon_bv, recon_bj, recon_cdr3b, 
                                               recon_av, recon_aj, recon_cdr3a,
                                               bv, bj, cdr3b, av, aj, cdr3a)
                kl_loss = KL_Div(mu, var)
               
                pcc.append(rna_corr_coef(recon_rna, rna))
                acc.append(tcr_recon_evaluate(recon_bv, recon_bj, recon_cdr3b, 
                                               recon_av, recon_aj, recon_cdr3a,
                                               bv, bj, cdr3b, av, aj, cdr3a))
                
                if epoch < warmup:
                    kl_weights = (epoch+1) / warmup
                else:
                    kl_weights=1
                
                if penalty == 'L2_dist':
                    penalty_loss = euclidean_dist(mu_R, mu_T)
                elif penalty == 'symmKL':
                    penalty_loss = symmKL(mu_R, var_R, mu_T, var_T)
                    # penalty_loss = 2 * kl_weights * penalty_loss
                elif penalty == 'mmd_rbf':
                    penalty_loss = mmd_rbf(zR, zT)

                loss = {'scRNA_recon_loss': scRNA_recon_loss, 
                        'scTCR_recon_loss': scTCR_recon_loss, 
                        'kl_loss': kl_weights * kl_loss,
                        'penalty_loss':penalty_loss}
                
                optimizer.zero_grad()
                sum(loss.values()).backward()
                optimizer.step()
                    
                for key in loss.keys():
                    epoch_loss[key] += loss[key].item()
                    
            epoch_loss = {key:value/(idx+1) for key, value in epoch_loss.items()}
            info = ','.join(['{}={:.3f}'.format(key, value) for key, value in epoch_loss.items()])
            info += ',rna_pcc={:.3f}'.format(np.mean(pcc))
            info += ',bv_acc={:.3f}'.format(np.mean([v[0] for v in acc]))
            info += ',bj_acc={:.3f}'.format(np.mean([v[1] for v in acc]))
            info += ',cdr3b_acc={:.3f}'.format(np.mean([v[2] for v in acc]))
            info += ',av_acc={:.3f}'.format(np.mean([v[3] for v in acc]))
            info += ',aj_acc={:.3f}'.format(np.mean([v[4] for v in acc]))
            info += ',cdr3a_acc={:.3f}'.format(np.mean([v[5] for v in acc]))
            t.set_postfix_str(info)
            
            if valid_dataloader is None:
                valid_loss = sum(epoch_loss.values())
            else:
                valid_loss = self.evalute(valid_dataloader,  penalty, kl_weights, device)
            early_stopping(valid_loss, self)
            if early_stopping.early_stop:
                print('EarlyStopping: run {} epoch'.format(epoch+1))
                break
            
    def evalute(self, valid_dataloader, penalty, kl_weights, device='cuda'):
        self.eval()
        epoch_loss = {'scRNA_recon_loss': 0.0, 'scTCR_recon_loss': 0.0, 
                        'kl_loss': 0.0,'penalty_loss':0.0}
        for idx, (rna, tcr, domain_label, _) in enumerate(valid_dataloader):
            rna = rna.double().to(device)
            tcr = tcr.long().to(device)
            bv = tcr[:,0].squeeze(-1)
            bj = tcr[:,1].squeeze(-1)
            cdr3b = tcr[:, 2:(2 + self.max_len)]
            av = tcr[:, 2 + self.max_len].squeeze(-1)
            aj = tcr[:, 3 + self.max_len].squeeze(-1)
            cdr3a = tcr[:, (4 + self.max_len):]              
                                                    
            zR, mu_R, var_R = self.RNAencoder(rna)
            zT, mu_T, var_T = self.TCRencoder(bv, bj, cdr3b,
                                                av, aj, cdr3a)
            z, mu, var = self.reparameterize(mu_R, mu_T, var_R, var_T,
                                            weights_mu=self.weights_mu, weights_var=self.weights_var)
            
            recon_rna = self.RNAdecoder(z, domain_label)
            recon_bv, recon_bj, recon_cdr3b, recon_av, recon_aj, recon_cdr3a = self.TCRdecoder(z)
            
            scRNA_recon_loss = scRNA_recon(recon_rna, rna)
            scTCR_recon_loss = scTCR_recon(recon_bv, recon_bj, recon_cdr3b, 
                                            recon_av, recon_aj, recon_cdr3a,
                                            bv, bj, cdr3b, av, aj, cdr3a)
            kl_loss = KL_Div(mu, var)
            
            if penalty == 'L2_dist':
                penalty_loss = euclidean_dist(mu_R, mu_T)
            elif penalty == 'symmKL':
                penalty_loss = symmKL(mu_R, var_R, mu_T, var_T)
                # penalty_loss = 2 * kl_weights * penalty_loss
            elif penalty == 'mmd_rbf':
                penalty_loss = mmd_rbf(zR, zT)

            loss = {'scRNA_recon_loss': scRNA_recon_loss, 
                    'scTCR_recon_loss': scTCR_recon_loss, 
                    'kl_loss': kl_weights * kl_loss,
                    'penalty_loss':penalty_loss}
            for key in loss.keys():
                epoch_loss[key] += loss[key].item()
        epoch_loss = {key:value/(idx+1) for key, value in epoch_loss.items()}
        return sum(epoch_loss.values())       

    def mix_multi(self, Xs, weights=None):
        Xs = torch.stack(Xs, dim=1)
        # Apply weights if provided
        if weights is not None:
            weights = F.softmax(weights, 0)
            weights = weights.unsqueeze(0).unsqueeze(-1) # (1 x 2 x 1)
            Xs = Xs * weights
            Xs = Xs.sum(dim=1)
        else :
            Xs = Xs.mean(dim=1)
        return Xs
    
    def reparameterize(self, mu_R, mu_T, var_R, var_T, weights_mu=None, weights_var=None):
        mu, var = self.mix_multi((mu_R, mu_T), weights_mu), self.mix_multi((var_R, var_T),weights_var)
        z = Normal(mu, var.sqrt()).rsample()
        return z, mu, var
    
    def load_model(self, path):
        """
        Load pre-trained model parameters from a file.

        Args:
            path (str): The file path to load the model from.
        """
        state_dict1 = torch.load(path)
        state_dict2 = self.state_dict()
        for key in state_dict1:
            if key in state_dict2:
                state_dict2[key] = state_dict1[key]
        self.load_state_dict(state_dict2)
        
    def gene_attn_weight(self, adata, n_samples=64, device='cuda'):
        self.to(device)
        self.eval()
        attn_weight_init = np.zeros((self.x_dims, self.x_dims))
        scdata = scRNADataset(adata)
        dataloader = DataLoaderX(scdata,  batch_size=n_samples, drop_last=False, shuffle=False, num_workers=8)
        for x, _, _ in dataloader:
            x = x.double().to(device)
            _,attn_weight = self.RNAencoder.self(x, output_attentions=True)
            attn_weight = attn_weight.detach().cpu()
            attn_weight = attn_weight.mean((0,1)).numpy()
            attn_weight_init += attn_weight
        attn_weight_init /= len(dataloader)
        np.fill_diagonal(attn_weight_init, 0)
        return pd.DataFrame(attn_weight_init,index=adata.var.index.tolist(),columns=adata.var.index.tolist())
    
    def celltype_attn_weight(self, adata, Celltype, n_samples=64, device='cuda'):
        self.to(device)
        self.eval()
        outs = pd.DataFrame(index=adata.var.index.tolist(),columns=adata.obs[Celltype].cat.categories)
        for celltype_value in adata.obs[Celltype].cat.categories:
            adata_Subs = adata[adata.obs[Celltype].isin([celltype_value])]
            attn_weight_init = np.zeros((self.x_dims))
            scdata = scRNADataset(adata_Subs)
            dataloader = DataLoaderX(scdata,  batch_size=n_samples, drop_last=False, shuffle=False, num_workers=8)
            for x, _, _ in dataloader:
                x = x.double().to(device)
                _,attn_weight = self.RNAencoder.self(x, output_attentions=True)
                attn_weight = attn_weight.detach().cpu().mean((0,1)).numpy()
                attn_weight_init += attn_weight.mean(0)
            attn_weight_init /= len(dataloader)
            outs[celltype_value] = attn_weight_init
        return outs
    
    def aa_attn_weight(self, adata, TCR_dict, batch_size, device='cuda'):
        self.to(device)
        self.eval()
        scdata = scTCRDataset(adata,TCR_dict=TCR_dict)
        dataloader = DataLoaderX(scdata,  batch_size=batch_size, drop_last=False, shuffle=False, num_workers=8)
        cdr3b_attn_mean = np.zeros((self.max_len, self.max_len))
        cdr3a_attn_mean = np.zeros((self.max_len, self.max_len))
        for idx, (tcr, _) in enumerate(dataloader):
            tcr = tcr.long().to(device)
            cdr3b = tcr[:, 2:(2 + self.max_len)]
            cdr3a = tcr[:, (4 + self.max_len):]
            attn_mask_b = (cdr3b == 0).clone().detach()
            attn_mask_a = (cdr3a == 0).clone().detach()
            
            cdr3b, cdr3a = self.TCRencoder.aa_embedding(cdr3b), self.TCRencoder.aa_embedding(cdr3a)
            _,attn_weight_b = self.TCRencoder.cdr3b_encode.self(cdr3b,cdr3b,cdr3b, 
                                                                attn_mask=None, key_padding_mask=attn_mask_b)
            _,attn_weight_a = self.TCRencoder.cdr3a_encode.self(cdr3a,cdr3a,cdr3a, 
                                                                attn_mask=None, key_padding_mask=attn_mask_a)
        
            attn_weight_b = attn_weight_b.detach().cpu().numpy().mean(0)
            attn_weight_a = attn_weight_a.detach().cpu().numpy().mean(0)
            cdr3b_attn_mean += attn_weight_b
            cdr3a_attn_mean += attn_weight_a
        np.fill_diagonal(cdr3b_attn_mean, 0)
        np.fill_diagonal(cdr3a_attn_mean, 0)
        cdr3b_df = pd.DataFrame(cdr3b_attn_mean/ (idx+1), index=list(range(1, self.max_len+1)), columns=list(range(1, self.max_len+1)))
        cdr3a_df = pd.DataFrame(cdr3a_attn_mean/ (idx+1), index=list(range(1, self.max_len+1)), columns=list(range(1, self.max_len+1)))
        return cdr3b_df, cdr3a_df
    
    def aa_embed(self, tensor, device):
        self.to(device)
        self.eval()
        tensor = tensor.to(device)
        outs = self.TCRencoder.aa_embedding.embedding(tensor)
        outs = outs.detach().squeeze(0).cpu().numpy()
        return outs
