import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

import torch
import os
import scanpy as sc
from typing import List

from .data import load_data, neighbors_umap
from .model.model import VAE_scRNA, VAE_scTCR, VAE_Multi
from .model.utils import seed_everything, make_TCR_dict

def MIST(rna_path:List[str]=None, 
    tcr_path:List[str]=None, 
    batch:List[str]=None,
    rna_data_type:str='h5ad',
    tcr_data_type:str='10X',
    protein_path:str=None,
    type:str='multi',         
    min_genes:int=600, 
    min_cells:int=3, 
    pct_mt:int=None,
    n_top_genes:int=2000, 
    backed:bool=False,
    batch_scale:bool=False,
    batch_min:int=0,
    remove_TCRGene:bool=False,
    max_len:int=30,
    scirpy:bool=True,
    batch_size:int=128,
    num_workers:int=8,
    aa_dims:int=64,
    gene_dims:int=48,
    pooling_dims:int=16,
    z_dims:int=128,
    drop_prob:float=0.1,
    weights:bool=False,
    lr:float=1e-4,
    weight_decay:float=1e-3,
    max_epoch:int=400, 
    patience:int=40, 
    warmup:int=40,
    penalty:str='mmd_rbf',
    gpu:int=0, 
    seed:int=42,
    outdir:str=None
    ):
    """
    """
    seed_everything(seed)
        
    if torch.cuda.is_available(): # cuda device
        device = 'cuda'
        torch.cuda.set_device(gpu)
    else:
        device = 'cpu'
        
    TCR_dict = make_TCR_dict()
    adata, *dataloader_tuple = load_data(rna_path=rna_path, 
                                tcr_path=tcr_path, 
                                batch=batch,
                                rna_data_type=rna_data_type,
                                tcr_data_type=tcr_data_type,
                                protein_path=protein_path,
                                type=type,         
                                min_genes=min_genes, 
                                min_cells=min_cells, 
                                pct_mt=pct_mt,
                                n_top_genes=n_top_genes, 
                                backed=backed,
                                batch_scale=batch_scale,
                                batch_min=batch_min,
                                remove_TCRGene=remove_TCRGene,
                                TCR_dict=TCR_dict,
                                max_len=max_len,
                                scirpy=scirpy,
                                batch_size=batch_size,
                                num_workers=num_workers
                                )
    print('Model training')
    if type == 'rna':
        model = VAE_scRNA(x_dims=adata.shape[1], z_dims=pooling_dims,
                          batchs=len(adata.obs['batch'].cat.categories)).double()
        
        model.fit(train_dataloader=dataloader_tuple[0],  valid_dataloader=dataloader_tuple[1],
                lr=lr, weight_decay=weight_decay, max_epoch=max_epoch, device=device, 
                patience=patience, outdir=os.path.join(outdir, 'model.pt') if outdir else 'model.pt')
        
    elif type == 'tcr':
        model = VAE_scTCR(z_dims=z_dims, aa_size=21, aa_dims=aa_dims, max_len=max_len, 
               bv_size=len(TCR_dict['TRBV']), bj_size=len(TCR_dict['TRBJ']),
               av_size=len(TCR_dict['TRAV']), aj_size=len(TCR_dict['TRAJ']), 
               gene_dims=gene_dims, drop_prob=drop_prob).double()
        
        model.fit(train_dataloader=dataloader_tuple[0], valid_dataloader=dataloader_tuple[1], 
          lr=lr, weight_decay=weight_decay, max_epoch=max_epoch, device=device, patience=patience, 
          outdir=os.path.join(outdir, 'model.pt') if outdir else 'model.pt')
        
    elif type == 'multi':
        model = VAE_Multi(x_dims=adata.shape[1], pooling_dims=pooling_dims,
                    z_dims=z_dims, batchs=len(adata.obs['batch'].cat.categories),
                    aa_size=21, aa_dims=aa_dims, max_len=max_len, 
                    bv_size=len(TCR_dict['TRBV']), bj_size=len(TCR_dict['TRBJ']),
                    av_size=len(TCR_dict['TRAV']), aj_size=len(TCR_dict['TRAJ']), 
                    gene_dims=gene_dims, drop_prob=drop_prob, weights=weights).double()
        
        model.fit(train_dataloader=dataloader_tuple[0], valid_dataloader=dataloader_tuple[1], 
              lr=lr, weight_decay=weight_decay, max_epoch=max_epoch, device=device, 
              penalty=penalty, patience=patience, warmup=warmup,
              outdir=os.path.join(outdir, 'model.pt') if outdir else 'model.pt')

    model.load_model(os.path.join(outdir, 'model.pt') if outdir else 'model.pt')
    print('Encode latent')
    if type == 'multi':
        # multi
        adata.obsm['latent'] = model._encodeMulti(dataloader_tuple[2], mode='latent', eval=True, 
                                                      device=device, TCR_dict=TCR_dict, temperature=1)
        pca_multi = PCA(n_components=15, random_state=seed)
        adata.obsm['latent_pca']=pca_multi.fit_transform(adata.obsm['latent'])
        
        # rna
        adata.obsm['latent_rna'] = model._encodeRNA(dataloader_tuple[3], mode='latent', eval=True, device=device)
        pca_rna = PCA(n_components=15, random_state=seed)
        adata.obsm['latent_rna_pca']=pca_rna.fit_transform(adata.obsm['latent_rna'])
        
        #tcr
        adata.obsm['latent_tcr'] = model._encodeTCR(dataloader_tuple[4], mode='latent', eval=True, device=device,
                                                        TCR_dict=TCR_dict, temperature=1)
        pca_tcr = PCA(n_components=15, random_state=seed)
        adata.obsm['latent_tcr_pca']=pca_tcr.fit_transform(adata.obsm['latent_tcr'])
        
        print('Clusting')
        neighbors_umap(adata, use_rep='latent_pca', n_pcs=None, key_added='multi')
        sc.tl.leiden(adata, resolution=1.0, neighbors_key='multi',key_added='multi-cluster')
        neighbors_umap(adata, use_rep='latent_rna_pca', n_pcs=None, key_added='rna')
        sc.tl.leiden(adata, resolution=1.0, neighbors_key='rna', key_added='rna-cluster')
        neighbors_umap(adata, use_rep='latent_tcr_pca', n_pcs=None, key_added='tcr')

    elif type == 'rna':
        adata.obsm['latent_rna'] = model._encode(dataloader_tuple[2], mode='latent', eval='True', device=device)
        
        print('Clusting')
        neighbors_umap(adata, use_rep='latent_rna', n_pcs=None, key_added='rna')
        sc.tl.leiden(adata, resolution=1.0, neighbors_key='rna', key_added='rna-cluster')

    elif type == 'tcr':
        adata.obsm['latent_tcr'] = model._encode(dataloader_tuple[2], mode='latent', eval=True, device=device,
                                                TCR_dict=TCR_dict, temperature=1)
        pca_tcr = PCA(n_components=15, random_state=seed)
        adata.obsm['latent_tcr_pca']=pca_tcr.fit_transform(adata.obsm['latent_tcr'])
        neighbors_umap(adata, use_rep='latent_tcr_pca', n_pcs=None, key_added='tcr')
        
    adata.write(os.path.join(outdir, 'adata.h5ad') if outdir else 'adata.h5ad')
    return adata, model

def remove_redundant_genes(df, min_occurrence=0.5, top=100):
    """_summary_

    Args:
        df (pd dataframe): df of attention weigth
        min_occurrence (float, optional): _description_. suguest 0.5 for celltype 0.2 for gene
        top (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """
    top_genes = df.apply(lambda x: x.sort_values(ascending=False)[:top].index.tolist(), axis=0)
    flattened_genes = pd.Series(top_genes.values.flatten())
    gene_counts = flattened_genes.value_counts()
    
    redundant_genes = gene_counts[gene_counts >= min_occurrence * len(df.columns)].index.tolist()
    df = df.drop(redundant_genes)
    return df

def label_transfer(adata_ref, adata_query, rep='latent', label='Celltype'):
    x1, y1 = adata_ref.obsm[rep], adata_ref.obs[label]
    x2 = adata_query.obsm[rep]
    
    knn = KNeighborsClassifier().fit(x1, y1)
    y2 = knn.predict(x2)
    return y2

def label_tcr_similarity(adata, label, latent,  similarity='cosine'):
    levels = adata.obs[label].cat.categories
    cos_array = np.empty((len(levels), len(levels)))
    
    if similarity=='cosine':
        tcr_similarity = cosine_similarity(
            adata.obsm[latent],
        adata.obsm[latent]
        )
    elif similarity=='L2dist':
        tcr_similarity = euclidean_distances(
            adata.obsm[latent]
            )
        tcr_similarity = tcr_similarity * -1

    for i, level_i in enumerate(levels):
        mask_i = adata.obs[label] == level_i
        for j, level_j in enumerate(levels):
            mask_j = adata.obs[label] == level_j
            similarity_ij = tcr_similarity[mask_i][:, mask_j]
            cos_array[i, j] = np.max(similarity_ij, axis=1).mean()
            
    # np.fill_diagonal(cos_array, np.nan)
    cos_array_scaled = MinMaxScaler().fit_transform(cos_array.T)
    df = pd.DataFrame(cos_array_scaled.T, index=levels, columns=levels)
    return df

def label_tcr_dist(adata, latent, label, similarity='L2dist'):
    levels = adata.obs[label].cat.categories
    dist_dict = dict()
    
    if similarity=='cosine':
        tcr_similarity = cosine_similarity(
            adata.obsm[latent],
        adata.obsm[latent]
        )
        tcr_similarity = 1-tcr_similarity
    elif similarity=='L2dist':
        tcr_similarity = euclidean_distances(
            adata.obsm[latent]
            )
        
    for i, level_i in enumerate(levels):
        mask_i = adata.obs[label] == level_i
        similarity_i = tcr_similarity[mask_i][:, mask_i]
        np.fill_diagonal(similarity_i, np.nan)
        similarity_i = similarity_i[np.triu_indices_from(similarity_i)]
        similarity_i = similarity_i[~np.isnan(similarity_i)]
        dist_dict[level_i] = similarity_i
    return dist_dict
