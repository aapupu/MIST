import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr

import anndata
import scanpy as sc
import scirpy as ir
from sklearn.preprocessing import MaxAbsScaler

from .model.dataset import scRNADataset, scTCRDataset, MultiDataset, DataLoaderX

def load_scTCR(path, batch, type='10X'):
    assert type in ['10X', 'tracer', 'BD', 'h5ad'], 'type of scTCR-seq data must in 10X, tracer, BD, or h5ad.'
    adatas_tcr = {}
    
    for sample, sample_meta in zip(path, batch):
        if type == '10X':
            adata_tcr = ir.io.read_10x_vdj(sample)
        elif type == 'tracer':
            adata_tcr = ir.io.read_tracer(sample)
        elif type == 'BD':
            adata_tcr = ir.io.read_bd_rhapsody(sample)
        elif type == 'h5ad':
            adata_tcr = ir.io.read_h5ad(sample)
        adata_tcr.obs['batch'] = sample_meta
        adatas_tcr[sample_meta] = adata_tcr

    adata_tcr = anndata.concat(adatas_tcr, index_unique="_")
    return adata_tcr
    
def load_scRNA(path, batch, type='h5ad'):
    assert type in ['10X mtx', 'h5', 'h5ad'], 'type of scRNA-seq data must in 10X mtx, h5, or h5ad.'
    adatas = {}
    
    for sample, sample_meta in zip(path, batch):
        if type == '10X mtx':
            adata = sc.read_10x_mtx(sample, var_names='gene_symbols', cache=True)
        elif type == 'h5':
            adata = sc.read_10x_h5(sample)
        elif type == 'h5ad':
            adata = sc.read_h5ad(sample)
        adata.var_names_make_unique()
        adata.X = scipy.sparse.csr_matrix(adata.X)
        adata.obs['batch'] = sample_meta
        adatas[sample_meta] = adata
        
    adatas = anndata.concat(adatas, index_unique='_')
    return adatas

def clr_function(x):
    x_positive = x[x > 0]
    if len(x_positive) == 0:
        return np.nan
    else:
        return np.log1p(x / (np.exp(np.sum(np.log1p(x_positive)) / len(x))))

def CLR(adata, axis=0):
    X_norm = np.apply_along_axis(clr_function, axis=axis, arr=np.array(adata.X.todense()))
    X_norm = scipy.sparse.csr_matrix(X_norm)
    adata.X = X_norm
    return adata

def process_scRNA(
        adata, 
        min_genes = 600, 
        min_cells = 3, 
        pct_mt = None,
        n_top_genes = 2000, 
        backed = False,
        multimodality=False,
        adata_tcr = None,
        protein=None,
        batch_scale=False,
        batch_min = 100,
        remove_TCRGene = False 
        
    ):
    print('Processing scRNA-seq data')
    if type(adata.X) != csr.csr_matrix and (not backed) and (not adata.isbacked):
            adata.X = scipy.sparse.csr_matrix(adata.X)
            
    print('Filtering cells and genes')
    adata = adata[:, [gene for gene in adata.var_names if not str(gene).startswith('ERCC')]]
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    if pct_mt:
        adata.var['mt'] = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        adata = adata[adata.obs.pct_counts_mt < pct_mt, :]
    else:
        adata = adata[:, [gene for gene in adata.var_names if not str(gene).startswith(tuple(['MT-', 'mt-']))]]
    
    if multimodality:
        print('Concat RNA and TCR')
        common_cells = adata.obs_names.intersection(adata_tcr.obs_names)
        adata = adata[common_cells]
        adata_tcr = adata_tcr[common_cells]
        assert (adata.obs.index == adata_tcr.obs.index).all()
        adata.obs = adata.obs.combine_first(adata_tcr.obs)
    
    if batch_min is not None:    
        adata.obs['batch'] = adata.obs['batch'].astype('str')
        batch_counts = adata.obs['batch'].value_counts()
        selected_batches = batch_counts[batch_counts >= batch_min].index
        adata = adata[adata.obs['batch'].isin(selected_batches)]
    
    print('Normalize and log1p per cell')
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    if remove_TCRGene:
        adata.var['Features'] = adata.var_names
        adata = adata[:,~adata.var['Features'].str.contains('^TRAV|^TRAJ|^TRBV|^TRBJ|^TRDV|^TRGV|^TRDJ|^TRGJ|^IGK|^IGH|^IGL')]
        
    if not pd.api.types.is_categorical_dtype(adata.obs['batch']):
        adata.obs['batch'] = adata.obs['batch'].astype('category')
        
    print('Finding HVG and maxabs scaling')
    if type(n_top_genes) == int:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key='batch',inplace=False, subset=True)
        adata = adata[:, adata.var.highly_variable]
    elif type(n_top_genes) == list:
        genes_list = [g for g in n_top_genes if g in adata.var_names]
        adata = adata[:, genes_list]       

    if protein:
        protein = CLR(protein[adata.obs.index])
        protein.raw = protein
        protein.var['feature_types'], adata.var['feature_types'] = 'ADT', 'Gene Expression'
        adata = anndata.concat([adata, protein], axis=1, merge='first')
        
    if batch_scale:
        for b in adata.obs['batch'].unique():
            idx = np.where(adata.obs['batch']==b)[0]
            maxabsscaler = MaxAbsScaler(copy=False).fit(adata.X[idx])
            adata.X[idx] = maxabsscaler.transform(adata.X[idx])    
    else: 
        adata.X = MaxAbsScaler().fit_transform(adata.X)
        
    print('Processed adata shape: {}'.format(adata.shape))
    return adata

def process_scTCR(
        adata_tcr, 
        max_len=30,
        TCR_dict=None,
        scirpy=None
    ):
    print('Filtering TCR')
    if scirpy: 
        ir.tl.chain_qc(adata_tcr)
        adata_tcr = adata_tcr[adata_tcr.obs['chain_pairing'].isin(['single pair']),:]
    
    adata_tcr = adata_tcr[adata_tcr.obs['IR_VDJ_1_v_call'].isin(TCR_dict['TRBV'].keys()) &
                        adata_tcr.obs['IR_VDJ_1_j_call'].isin(TCR_dict['TRBJ'].keys()) & 
                        adata_tcr.obs['IR_VJ_1_v_call'].isin(TCR_dict['TRAV'].keys()) & 
                        adata_tcr.obs['IR_VJ_1_j_call'].isin(TCR_dict['TRAJ'].keys()),:]
    
    adata_tcr = adata_tcr[(~adata_tcr.obs['IR_VDJ_1_junction_aa'].str.contains(r"[\\X|\\-|\\+|\\*]",na=True)) &
                          (~adata_tcr.obs['IR_VJ_1_junction_aa'].str.contains(r"[\\X|\\-|\\+|\\*]",na=True)) &
                          adata_tcr.obs['IR_VDJ_1_junction_aa'].str.len().le(max_len) &
                          adata_tcr.obs['IR_VJ_1_junction_aa'].str.len().le(max_len),:]
    adata_tcr.obs = adata_tcr.obs.dropna(axis=1, how='all')
    
    print('Processed scTCR-seq shape: {}'.format(adata_tcr.shape))
    return adata_tcr

def process_multimode(
        adata, 
        adata_tcr=None, 
        max_len=30,
        TCR_dict=None,
        scirpy=True,
        min_genes = 600, 
        min_cells = 3, 
        pct_mt = None,
        n_top_genes = 2000, 
        backed = False,
        protein=None,
        batch_scale=False,
        batch_min=0,
        remove_TCRGene = False 
    ):
    if adata_tcr:
        adata_tcr = process_scTCR(adata_tcr, max_len, TCR_dict, scirpy)
        adata = process_scRNA(adata, min_genes, min_cells, pct_mt, n_top_genes, backed, 
                        multimodality=True, adata_tcr = adata_tcr,protein=protein,batch_scale=batch_scale,
                        batch_min=batch_min, remove_TCRGene=remove_TCRGene)
    else:
        adata_tcr = process_scTCR(adata, max_len, TCR_dict)
        adata = process_scRNA(adata, min_genes, min_cells, pct_mt, n_top_genes, backed, 
                        multimodality=True, adata_tcr=adata_tcr, protein=protein, 
                        batch_scale=batch_scale, batch_min=batch_min, remove_TCRGene=remove_TCRGene)
    return adata

def neighbors_umap(adata, use_rep=None, n_pcs=None, key_added=None):
    sc.pp.neighbors(adata, n_neighbors=30, use_rep=use_rep, n_pcs=n_pcs, key_added=key_added)
    sc.tl.umap(adata, min_dist=0.1, neighbors_key=key_added)
    adata.obsm[('X_'+key_added+'_umap')] = adata.obsm['X_umap']

def load_data(rna_path=None, 
            tcr_path=None, 
            batch=None,
            rna_data_type=None,
            tcr_data_type=None,
            protein_path=None,
            type='multi',         
            min_genes=600, 
            min_cells=3, 
            pct_mt=None,
            n_top_genes=2000, 
            backed=False,
            batch_scale=False,
            batch_min=0,
            remove_TCRGene=False,
            TCR_dict=None,
            max_len=30,
            scirpy=True,
            batch_size=128,
            num_workers=8
             ):
                 
    if rna_path is not None:
        if batch is None:
            batch = [str(i) for i in range(len(rna_path))]
        adata = load_scRNA(rna_path, batch, rna_data_type)
        print('Raw adata shape: {}'.format(adata.shape))
    else:
        adata = None
        
    if tcr_path is not None: 
        if batch is None:
            batch = [str(i) for i in range(len(tcr_path))]
        adata_tcr = load_scTCR(tcr_path, batch, tcr_data_type)
        adata_tcr.obs['IR_VJ_1_v_call'] = adata_tcr.obs['IR_VJ_1_v_call'].astype('str')
        condition = adata_tcr.obs['IR_VJ_1_v_call'].str.contains('DV') & ~adata_tcr.obs['IR_VJ_1_v_call'].str.contains('/DV')
        if any(condition):
            adata_tcr.obs['IR_VJ_1_v_call'] = adata_tcr.obs['IR_VJ_1_v_call'].apply(lambda x: x.replace('DV', '/DV').replace('OR', '/OR') if ('DV' in x) or ('OR' in x) else x)
        print('Raw adata_tcr shape: {}'.format(adata_tcr.shape))
    else:
        adata_tcr = None
        
    if protein_path is not None:
        protein = sc.read_h5ad(protein_path)
        print('Raw adata_protein shape: {}'.format(protein.shape))
    else:
        protein = None
        
    assert type in ['rna', 'tcr', 'multi'], 'type must in rna, tcr or multi.'
    if type == 'rna':
        adata = process_scRNA(adata, 
                        min_genes=min_genes, 
                        min_cells=min_cells, 
                        pct_mt=pct_mt,
                        n_top_genes=n_top_genes, 
                        backed=False,
                        multimodality=False,
                        adata_tcr=None,
                        protein=protein,
                        batch_scale=batch_scale,
                        batch_min=batch_min,
                        remove_TCRGene=remove_TCRGene)
        sc.tl.pca(adata, svd_solver='arpack')
        neighbors_umap(adata, use_rep=None, n_pcs=20, key_added='raw')
        scdata = scRNADataset(adata)
        
    elif type == 'tcr':
        adata = process_scTCR(adata_tcr, 
                            max_len=max_len,
                            TCR_dict=TCR_dict,
                            scirpy=scirpy)
        scdata = scTCRDataset(adata,TCR_dict)
        
    elif type == 'multi':
        adata = process_multimode(adata, 
                            adata_tcr=adata_tcr, 
                            max_len=max_len,
                            TCR_dict=TCR_dict,
                            scirpy=scirpy,
                            min_genes=min_genes, 
                            min_cells=min_cells, 
                            pct_mt=pct_mt,
                            n_top_genes=n_top_genes, 
                            backed=backed,
                            protein=protein,
                            batch_scale=batch_scale,
                            batch_min=batch_min,
                            remove_TCRGene=remove_TCRGene)
        sc.tl.pca(adata, svd_solver='arpack')
        neighbors_umap(adata, use_rep=None, n_pcs=20, key_added='raw')
        scdata = MultiDataset(adata,TCR_dict)
        rna_scdata = scRNADataset(adata)
        tcr_scdata = scTCRDataset(adata, TCR_dict)

    train_set, val_set = torch.utils.data.random_split(scdata, [0.9, 0.1])    
    trainloader = DataLoaderX(train_set, batch_size=batch_size, 
                            drop_last=True, shuffle=True, num_workers=num_workers, pin_memory=True)
    validloader = DataLoaderX(val_set, batch_size=batch_size, 
                            drop_last=True, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = DataLoaderX(scdata,  batch_size=batch_size, 
                            drop_last=False, shuffle=False, num_workers=num_workers, pin_memory=True)
    if type == 'multi':
        rna_testdataloder = DataLoaderX(rna_scdata, batch_size=batch_size, 
                                    drop_last=False, shuffle=False, num_workers=num_workers, pin_memory=True)
        tcr_testdataloder = DataLoaderX(tcr_scdata, batch_size=batch_size, 
                                    drop_last=False, shuffle=False, num_workers=num_workers, pin_memory=True)
        return adata, trainloader, validloader, testloader, rna_testdataloder, tcr_testdataloder
    else:
        return adata, trainloader, validloader, testloader 
