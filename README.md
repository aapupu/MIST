# Multi-InSight for T-cell（MIST）
MIST: Deep Learning Integration of Single-Cell RNA and TCR Sequencing Data for T-Cell Insight

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
pip install .
```

**Note: If cuda is available, GPU modes will be run automatically.**

Usage
-------
### 1. API function
```bash
from mist import MIST
adata, model = MIST(rna_path, tcr_path, batch, rna_data_type, tcr_data_type, type)
```

### 2. Command line
```bash
MIST --rna_path rna_path1 rna_path2 --tcr_path tcr_path1 tcr_path2 --batch batch1 batch2 --rna_data_type h5ad --tcr_data_type 10X --type multi
```

#### Output 
- adata.h5ad: preprocessed data and results
- model.pt: saved model

#### Option
- **--rna_path**<br />scRNA-seq data paths
- **--tcr_path**<br />scTCR-seq data paths
- **--batch**<br />batch imformation
- **--rna_data_type**<br />scRNA-seq data type
- **--tcr_data_type**<br />scTCR-seq data type
- **--protein_path**<br />ADT data path
- **--type**<br />Model type
- **--min_genes**<br />Filtered out cells that are detected in less than min_genes. Default: 600.
- **--min_cells**<br />Filtered out genes that are detected in less than min_cells. Default: 3.
- **--pct_mt**<br />Filtered out genes that are mt. Default: None.
- **--n_top_genes**<br />Number of highly-variable genes to keep. Default: 2000.
- **--batch_size**<br />Batch size
- **--pooling_dims**<br />Pooling_dims.
- **--z_dims**<br />latent dims. If type='rna', z_dims==pooling_dims.
- **--drop_prob**<br />Drop_prob of TCR.
- **--lr**<br />Learning rate.
- **--weight_decay**<br />Weight_decay of learning rate.
- **--max_epoch**<br />Max epochs for training. 
- **--patience**<br />Max epochs for easy-stop.
- **--warmup**<br />Epochs for warm up.
- **--gpu**<br />Index of GPU to use if GPU is available. Default: 0.
- **--seed**<br />Random seed.
- **--outdir**<br />Output directory.

Citation
-------
**MIST**

Contacts
-------
kyzy850520@163.com
