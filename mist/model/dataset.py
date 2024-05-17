import numpy as np
from prefetch_generator import BackgroundGenerator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .utils import tcr_to_vec


class scRNADataset(Dataset):
    """
    Dataset of scRNA-seq data.
    """
    def __init__(self, adata):
        """
        Initialize the scRNA-seq dataset.

        Args:
            adata: Annotated Data object containing scRNA-seq data.
        """        
        self.adata = adata
        self.shape = adata.shape
        
    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, idx):
        if isinstance(self.adata.X[idx], np.ndarray):
            x = self.adata.X[idx].squeeze().astype(float)
        else:
            x = self.adata.X[idx].toarray().squeeze().astype(float)
                
        domain_label = self.adata.obs['batch'].cat.codes.iloc[idx]
        return x, domain_label, idx

class scTCRDataset(Dataset):
    """
    Dataset of scTCR-seq data.
    """
    def __init__(self, adata, TCR_dict):
        """
        Initialize the scTCR-seq dataset.

        Args:
            adata: Annotated Data object containing scTCR-seq data.
            TCR_dict: Dictionary containing TCR gene information.
        """
        self.adata = adata
        self.shape = adata.shape
        self.TCR_dict = TCR_dict
        
    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, idx):
        bv, bj, cdr3b, av, aj, cdr3a =  tcr_to_vec(self.adata[idx], self.TCR_dict['AA'],
                                                self.TCR_dict['TRBV'], self.TCR_dict['TRBJ'], 
                                                self.TCR_dict['TRAV'], self.TCR_dict['TRAJ'])
        tcr = np.concatenate([bv.reshape(-1,1), bj.reshape(-1,1), cdr3b, 
                        av.reshape(-1,1), aj.reshape(-1,1), cdr3a], axis=1).squeeze()
        return tcr, idx

class MultiDataset(Dataset):
    """
    Dataset of scRNA-seq and scTCR-seq data.
    """
    def __init__(self, adata, TCR_dict):
        """
        Initialize the scRNA-seq and scTCR-seq dataset.

        Args:
            adata: Annotated Data object containing scRNA-seq and scTCR-seq data.
            TCR_dict: Dictionary containing TCR gene information.
        """
        self.adata = adata
        self.shape = adata.shape
        self.TCR_dict = TCR_dict

    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, idx):
        if isinstance(self.adata.X[idx], np.ndarray):
            x = self.adata.X[idx].squeeze().astype(float)
        else:
            x = self.adata.X[idx].toarray().squeeze().astype(float)

        domain_label = self.adata.obs['batch'].cat.codes.iloc[idx]
        bv, bj, cdr3b, av, aj, cdr3a = tcr_to_vec(self.adata[idx], self.TCR_dict['AA'],
                                                self.TCR_dict['TRBV'], self.TCR_dict['TRBJ'], 
                                                self.TCR_dict['TRAV'], self.TCR_dict['TRAJ']) 
        tcr = np.concatenate([bv.reshape(-1,1), bj.reshape(-1,1), cdr3b, 
                              av.reshape(-1,1), aj.reshape(-1,1), cdr3a], axis=1).squeeze()
        return x, tcr, domain_label, idx#, bv, bj, cdr3b, av, aj, cdr3a

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
