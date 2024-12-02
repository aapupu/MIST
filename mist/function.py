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
    type:str='joint',         
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
    max_epoch:int=300, 
    patience:int=30, 
    warmup:int=30,
    penalty:str='mmd_rbf',
    gpu:int=0, 
    seed:int=42,
    outdir:str=None
    ):
    """
    MIST: Deep Learning Integration of Single-Cell RNA and TCR Sequencing Data for T-Cell Insight.

    Parameters:
    - rna_path (List[str]): List of paths to scRNA-seq data files.
    - tcr_path (List[str]): List of paths to scTCR-seq data files.
    - batch (List[str]): List of batch labels.
    - rna_data_type (str): Type of scRNA-seq data file (e.g., 'h5ad').
    - tcr_data_type (str): Type of scTCR-seq data file (e.g., '10X').
    - protein_path (str): Path to merged protein (ADT) data file.
    - type (str): Type of model to train ('joint', 'rna', or 'tcr').
    - min_genes (int): Filtered out cells that are detected in less than min_genes. Default: 600.
    - min_cells (int): Filtered out genes that are detected in less than min_cells. Default: 3.
    - pct_mt (int): Filtered out cells that are detected in more than percentage of mitochondrial genes. If None, Filtered out mitochondrial genes. Default: None.
    - n_top_genes (int): Number of highly-variable genes to keep. Default: 2000.
    - backed (bool): Whether to use backed format for reading data. Default: False.
    - batch_scale (bool): Whether to data scale pre batch. Default: False.
    - batch_min (int): Filtered out batch that are detected in less than cells. Default: 0.
    - remove_TCRGene (bool): Whether to remove TCR gene (e.g., 'TRAV1-2'). Default: False.
    - max_len (int): Maximum length of cdr3aa sequence. Default: 30.
    - scirpy (bool): Whether to use scirpy package for filtering TCR. Default: True.
    - batch_size (int): Batch size for training.
    - num_workers (int): Number of workers for data loading. 
    - aa_dims (int): Dimensionality of amino acid embeddings. Default: 64.
    - gene_dims (int): Dimensionality of gene embeddings. Default: 48.
    - pooling_dims (int): Dimensionality of pooling layer. Default: 16.
    - z_dims (int): Dimensionality of latent space. Default: 128.
    - drop_prob (float): Dropout probability. Default: 0.1.
    - weights (bool): Whether to use weighted two mode latent space embedding. Default: False.
    - lr (float): Learning rate. Default: 1e-4.
    - weight_decay (float): Weight decay. Default: 1e-3.
    - max_epoch (int): Maximum number of epochs. Default: 300.
    - patience (int): Patience for early stopping. Default: 30.
    - warmup (int): Warmup epochs. Default: 30.
    - penalty (str): Type of penalty loss. Default: 'mmd_rbf'.
    - gpu (int): Index of GPU to use if GPU is available. Default: 0.
    - seed (int): Random seed.
    - outdir (str): Output directory.

    Returns:
    - adata: Preprocessed adata.
    - model: Trained model.
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
    # rna
    if type == 'rna':
        model = VAE_scRNA(x_dims=adata.shape[1], z_dims=pooling_dims,
                          batchs=len(adata.obs['batch'].cat.categories)).double()
        
        model.fit(train_dataloader=dataloader_tuple[0],  valid_dataloader=dataloader_tuple[1],
                lr=lr, weight_decay=weight_decay, max_epoch=max_epoch, device=device, 
                patience=patience, outdir=os.path.join(outdir, 'model.pt') if outdir else 'model.pt')
        
    # tcr    
    elif type == 'tcr':
        model = VAE_scTCR(z_dims=z_dims, aa_size=21, aa_dims=aa_dims, max_len=max_len, 
               bv_size=len(TCR_dict['TRBV']), bj_size=len(TCR_dict['TRBJ']),
               av_size=len(TCR_dict['TRAV']), aj_size=len(TCR_dict['TRAJ']), 
               gene_dims=gene_dims, drop_prob=drop_prob).double()
        
        model.fit(train_dataloader=dataloader_tuple[0], valid_dataloader=dataloader_tuple[1], 
          lr=lr, weight_decay=weight_decay, max_epoch=max_epoch, device=device, patience=patience, 
          outdir=os.path.join(outdir, 'model.pt') if outdir else 'model.pt')
        
    # joint    
    elif type == 'joint':
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
    if type == 'joint':
        # joint
        adata.obsm['latent_joint'] = model._encodeMulti(dataloader_tuple[2], mode='latent', eval=True, 
                                                      device=device, TCR_dict=TCR_dict, temperature=1)
        pca_joint = PCA(n_components=15, random_state=seed)
        adata.obsm['latent_joint_pca'] = pca_joint.fit_transform(adata.obsm['latent_joint'])
        
        # rna
        adata.obsm['latent_gex'] = model._encodeRNA(dataloader_tuple[3], mode='latent', eval=True, device=device)
        pca_rna = PCA(n_components=15, random_state=seed)
        adata.obsm['latent_gex_pca'] = pca_rna.fit_transform(adata.obsm['latent_gex'])
        
        # tcr
        adata.obsm['latent_tcr'] = model._encodeTCR(dataloader_tuple[4], mode='latent', eval=True, device=device,
                                                        TCR_dict=TCR_dict, temperature=1)
        pca_tcr = PCA(n_components=15, random_state=seed)
        adata.obsm['latent_tcr_pca'] = pca_tcr.fit_transform(adata.obsm['latent_tcr'])
        
        print('Clustering')
        neighbors_umap(adata, use_rep='latent_joint_pca', n_pcs=None, key_added='joint')
        sc.tl.leiden(adata, resolution=1.0, neighbors_key='joint', key_added='joint_cluster')
        neighbors_umap(adata, use_rep='latent_gex_pca', n_pcs=None, key_added='gex')
        sc.tl.leiden(adata, resolution=1.0, neighbors_key='gex', key_added='gex_cluster')
        neighbors_umap(adata, use_rep='latent_tcr_pca', n_pcs=None, key_added='tcr')

    elif type == 'rna':
        adata.obsm['latent_gex'] = model._encode(dataloader_tuple[2], mode='latent', eval='True', device=device)
        
        print('Clustering')
        neighbors_umap(adata, use_rep='latent_gex', n_pcs=None, key_added='gex')
        sc.tl.leiden(adata, resolution=1.0, neighbors_key='gex', key_added='gex_cluster')

    elif type == 'tcr':
        adata.obsm['latent_tcr'] = model._encode(dataloader_tuple[2], mode='latent', eval=True, device=device,
                                                TCR_dict=TCR_dict, temperature=1)
        pca_tcr = PCA(n_components=15, random_state=seed)
        adata.obsm['latent_tcr_pca']=pca_tcr.fit_transform(adata.obsm['latent_tcr'])
        neighbors_umap(adata, use_rep='latent_tcr_pca', n_pcs=None, key_added='tcr')
        
    adata.write(os.path.join(outdir, 'adata.h5ad') if outdir else 'adata.h5ad')
    return adata, model

def remove_redundant_genes(df, min_occurrence=0.5, top=100):
    """
    Remove redundant genes from a dataframe of attention weights.

    Args:
        df: DataFrame containing attention weights.
        min_occurrence: Minimum occurrence threshold for genes to be considered redundant.
                        Suggested value is 0.5 for cell type and 0.2 for gene.
        top : Number of top genes to consider in each column. Defaults to 100.

    Returns:
        pd.DataFrame: DataFrame with redundant genes removed.
    """
    top_genes = df.apply(lambda x: x.sort_values(ascending=False)[:top].index.tolist(), axis=0)
    flattened_genes = pd.Series(top_genes.values.flatten())
    gene_counts = flattened_genes.value_counts()
    
    redundant_genes = gene_counts[gene_counts >= min_occurrence * len(df.columns)].index.tolist()
    df = df.drop(redundant_genes)
    return df

def label_transfer(adata_ref, adata_query, rep='latent', label='Celltype'):
    """
    Transfer labels from one AnnData object to another based on a given representation.

    Args:
        adata_ref: Reference AnnData object containing the original labels.
        adata_query: Query AnnData object to which labels will be transferred.
        rep: Representation used for transferring labels. Defaults to 'latent'.
        label: Name of the label column in adata_ref. Defaults to 'Celltype'.

    Returns:
        np.ndarray: Transferred labels for adata_query.
    """
    x1, y1 = adata_ref.obsm[rep], adata_ref.obs[label]
    x2 = adata_query.obsm[rep]
    
    knn = KNeighborsClassifier().fit(x1, y1)
    y2 = knn.predict(x2)
    return y2

def label_tcr_similarity(adata, label, latent,  similarity='L2dist'):
    """
    Calculate pairwise similarity between TCR sequences by label based on a given representation.

    Args:
        adata: AnnData object containing TCR data.
        label: Name of the label column in adata.obs.
        latent: Name of the representation used for similarity calculation.
        similarity: Type of similarity metric to use. Defaults to 'L2dist'.

    Returns:
        pd.DataFrame: DataFrame containing pairwise similarity scores between TCR sequences.
    """
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
    """
    Calculate pairwise distances between TCR sequences of cluster based on a given representation.

    Args:
        adata: AnnData object containing TCR data.
        latent: Name of the representation used for distance calculation.
        label: Name of the label column in adata.obs.
        similarity: Type of distance metric to use. Defaults to 'L2dist'.

    Returns:
        dict: Dictionary containing pairwise distances for each label category.
    """
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

# from fuzzywuzzy import fuzz
# def compute_tcr_dist_sampling(adata, latent1, latent2, latent3, tcr, numbers=5000):
#     """
#     Computes the pairwise edit distances between TCR sequences and their corresponding L2 distances 
#     in three different latent spaces, based on random sampling.
    
#     Args:
#     adata : AnnData
#         The annotated data object containing TCR sequences in `obs` and latent representations in `obsm`.
#     latent1, latent2, latent3 : str
#         Keys in `adata.obsm` representing three latent spaces for comparison.
#     tcr : str
#         Key in `adata.obs` containing TCR sequences to compare.
#     numbers : int, optional, default=5000
#         Number of random pairs to sample for distance computation.

#     Returns:
#     l2_distances1, l2_distances2, l2_distances3 : np.ndarray
#         Arrays containing L2 distances between TCR pairs in the three latent spaces.
#     edit_distances : np.ndarray
#         Array containing the normalized edit distances (1 - partial similarity) between TCR pairs.
#     """

#     sequences = adata.obs[tcr]  
#     latent_vectors1 = adata.obsm[latent1] 
#     latent_vectors2 = adata.obsm[latent2]
#     latent_vectors3 = adata.obsm[latent3]
    
#     n = len(sequences)
#     if n < 2:
#         raise ValueError("Number of TCR sequences is less than 2")

#     # Initialize arrays to store computed distances
#     edit_distances = []
#     l2_distances1, l2_distances2, l2_distances3 = [], [], []

#     for _ in range(numbers):
#         # Randomly sample two indices without replacement
#         idx1, idx2 = random.sample(range(n), 2)
        
#         # Get the sequences corresponding to the sampled indices
#         seq1, seq2 = sequences[idx1], sequences[idx2]
        
#         # Compute normalized edit distance (1 - partial ratio similarity)
#         edit_distance = 1 - fuzz.partial_ratio(seq1, seq2) / 100.0
#         edit_distances.append(edit_distance)
        
#         # Compute L2 distances in the three latent spaces
#         l2_distance1 = euclidean_distances(
#             latent_vectors1[idx1].reshape(1, -1),
#             latent_vectors1[idx2].reshape(1, -1)
#         )[0, 0]
#         l2_distances1.append(l2_distance1)

#         l2_distance2 = euclidean_distances(
#             latent_vectors2[idx1].reshape(1, -1),
#             latent_vectors2[idx2].reshape(1, -1)
#         )[0, 0]
#         l2_distances2.append(l2_distance2)

#         l2_distance3 = euclidean_distances(
#             latent_vectors3[idx1].reshape(1, -1),
#             latent_vectors3[idx2].reshape(1, -1)
#         )[0, 0]
#         l2_distances3.append(l2_distance3)

#     edit_distances = np.array(edit_distances)
#     l2_distances1 = np.array(l2_distances1)
#     l2_distances2 = np.array(l2_distances2)
#     l2_distances3 = np.array(l2_distances3)
#     return l2_distances1, l2_distances2, l2_distances3, edit_distances
