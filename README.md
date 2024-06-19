# MIST (Multi-InSight for T cell)
MIST: an interpretable and flexible deep learning framework for single-T cell transcriptome and receptor analysis

<div align=center><img  height="600" src=https://github.com/aapupu/MIST/blob/main/docs/MIST.jpg><div align=left>

Installation
-------
### Install from PyPI
```bash
pip install mist
```

### Install from GitHub
install the latest develop version
```bash
pip install git+https://github.com/aapupu/MIST.git
```
or git clone and install
```bash
git clone git://github.com/aapupu/MIST.git
cd MIST
pip install -e .
```

**Note: MIST is implemented in Pytorch framework. If cuda is available, GPU modes will be run automatically.**

Usage
-------
### 1. API function
```bash
from mist import MIST
adata, model = MIST(rna_path, tcr_path, batch, rna_data_type, tcr_data_type, type)
```
Parameters of API function are similar to command line options.<br />
The output includes a trained model and an Anndata object, which can be further analyzed using scanpy and scirpy.<br />
```rna_path``` List of paths to scRNA-seq data files.<br />
```tcr_path``` List of paths to scTCR-seq data files.<br />
```batch``` List of batch labels.<br />
```rna_data_type``` Type of scRNA-seq data file (e.g., 'h5ad').<br />
```tcr_data_type``` Type of scTCR-seq data file (e.g., '10X').<br />
```type``` Type of model to train ('multi', 'rna', or 'tcr').

### 2. Command line
```bash
MIST --rna_path rna_path1 rna_path2 --tcr_path tcr_path1 tcr_path2 --batch batch1 batch2 --rna_data_type h5ad --tcr_data_type 10X --type multi
```

#### Output 
- adata.h5ad: preprocessed data and results
- model.pt: saved model

#### Option
- **--rna_path**<br />Paths to scRNA-seq data files.
- **--tcr_path**<br />Paths to scTCR-seq data files.
- **--batch**<br />Batch labels.
- **--rna_data_type**<br />Type of scRNA-seq data file.
- **--tcr_data_type**<br />Type of scTCR-seq data file.
- **--protein_path**<br />Path to merged protein (ADT) data file.
- **--type**<br />Type of model to train.
- **--min_genes**<br />Filtered out cells that are detected in less than min_genes. Default: 600.
- **--min_cells**<br />Filtered out genes that are detected in less than min_cells. Default: 3.
- **--pct_mt**<br />Filtered out cells that are detected in more than percentage of mitochondrial genes. If None, Filtered out mitochondrial genes. Default: None.
- **--n_top_genes**<br />Number of highly-variable genes to keep. Default: 2000.
- **--batch_size**<br />Batch size for training.
- **--pooling_dims**<br />Dimensionality of pooling layer. Default: 16.
- **--z_dims**<br />Dimensionality of latent space. If type='rna', z_dims=pooling_dims. Default: 128.
- **--drop_prob**<br />Dropout probability. Default: 0.1.
- **--lr**<br />Learning rate for the optimizer. Default: 1e-4.
- **--weight_decay**<br />L2 regularization strength. Default: 1e-3.
- **--max_epoch**<br />Maximum number of epochs. Default: 400.
- **--patience**<br />Patience for early stopping. Default: 40.
- **--warmup**<br />Warmup epochs. Default: 40.
- **--gpu**<br />Index of GPU to use if GPU is available. Default: 0.
- **--seed**<br />Random seed.
- **--outdir**<br />Output directory.

Citation
-------
**MIST**

Contacts
-------
kyzy850520@163.com
