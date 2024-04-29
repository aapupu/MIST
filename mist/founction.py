import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

import torch
import numpy as np
import os
import scanpy as sc
from anndata import AnnData
from typing import Union, List

from .data import load_data
from .model.model import VAE
from .model.utils import EarlyStopping
from .model.metrics import 

def MIST(adata):
    return adata

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
