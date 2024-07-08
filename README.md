# MIST (Multi-InSight for T cell)
MIST: an interpretable and flexible deep learning framework for single-T cell transcriptome and receptor analysis

<div align=center><img  height="500" src=https://github.com/aapupu/MIST/blob/main/doc/MIST.jpg><div align=left>

Installation
-------
### Install from PyPI
```bash
pip install mist-tcr
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

**Note: Python 3.8 is recommended. MIST is implemented in Pytorch framework. If cuda is available, GPU modes will be run automatically.**

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
```type``` Type of model to train ('joint', 'rna', or 'tcr').

### 2. Command line
```bash
MIST --rna_path rna_path1 rna_path2 --tcr_path tcr_path1 tcr_path2 --batch batch1 batch2 --rna_data_type h5ad --tcr_data_type 10X --type joint
```

#### Output 
- adata.h5ad: preprocessed data and results
- model.pt: saved model

#### Option
- **--rna_path**<br />Paths to scRNA-seq data files. (example: XXX1.h5ad XXX2.h5ad)
- **--tcr_path**<br />Paths to scTCR-seq data files. (example: XXX1.csv XXX2.csv)
- **--batch**<br />Batch labels. 
- **--rna_data_type**<br />Type of scRNA-seq data file (e.g., 10X mtx, h5, or h5ad).  Default: h5ad
- **--tcr_data_type**<br />Type of scTCR-seq data file (e.g., 10X, tracer, BD, or h5ad). Default: 10X
- **--protein_path**<br />Path to merged protein (ADT) data file.
- **--type**<br />Type of model to train (e.g., joint, rna, or tcr). Default: joint
- **--min_genes**<br />Filtered out cells that are detected in less than min_genes. Default: 600
- **--min_cells**<br />Filtered out genes that are detected in less than min_cells. Default: 3
- **--pct_mt**<br />Filtered out cells that are detected in more than percentage of mitochondrial genes. If None, Filtered out mitochondrial genes. Default: None
- **--n_top_genes**<br />Number of highly-variable genes to keep. Default: 2000
- **--batch_size**<br />Batch size for training. Default: 128
- **--pooling_dims**<br />Dimensionality of pooling layer. Default: 16
- **--z_dims**<br />Dimensionality of latent space. If type='rna', z_dims=pooling_dims. Default: 128
- **--drop_prob**<br />Dropout probability. Default: 0.1
- **--lr**<br />Learning rate for the optimizer. Default: 1e-4
- **--weight_decay**<br />L2 regularization strength. Default: 1e-3
- **--max_epoch**<br />Maximum number of epochs. Default: 300
- **--patience**<br />Patience for early stopping. Default: 30
- **--warmup**<br />Warmup epochs. Default: 30
- **--gpu**<br />Index of GPU to use if GPU is available. Default: 0
- **--seed**<br />Random seed. Default: 42
- **--outdir**<br />Output directory.

#### Help
Explore further applications of MIST
```bash
MIST.py --help 
```
The running examples of MIST can be found in the jupyter folder.

Citation
-------
**MIST: an interpretable and flexible deep learning framework for single-T cell transcriptome and receptor analysis**<br />
Wenpu Lai, Yangqiu Li, Oscar Junhong Luo<br />
bioRxiv 2024.07.05.602192; doi: https://doi.org/10.1101/2024.07.05.602192 

Contacts
-------
kyzy850520@163.com<br />
luojh@jnu.edu.cn
