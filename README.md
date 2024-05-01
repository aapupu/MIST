# MultiInSight T-cell（MIST）
MIST: Deep Learning Integration of Single-Cell RNA and TCR Sequencing Data for T-Cell Insight

![image](https://github.com/aapupu/MIST/blob/main/docs/MIST.png)

Installation
-------
### Install from PyPI
```bash
pip install mist
```

### Install from GitHub
install from PyPI
```bash
pip install mist
```
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
MIST --rna_path rna_path1 rna_path2 --tcr_path tcr_path1 tcr_path2 --batch batch1 batch1 --rna_data_type h5ad --tcr_data_type 10X --type multi
```

#### Output 
- adata.h5ad: preprocessed data and results
- model.pt: saved model

#### Option
- **--rna_path**<br />
scRNA-seq data paths
- --tcr_path
scTCR-seq data paths
- --batch
batch imformation
- --rna_data_type
scRNA-seq data type
- --tcr_data_type
scTCR-seq data type
- --protein_path
ADT data path
- --type
Model type
- --min_genes
Filtered out cells that are detected in less than min_genes. Default: 600.
- --min_cells
Filtered out genes that are detected in less than min_cells. Default: 3.
- --pct_mt
Filtered out genes that are mt. Default: None.
- --n_top_genes
Number of highly-variable genes to keep. Default: 2000.
- --batch_size
Batch size
- --pooling_dims
Pooling_dims.
- --z_dims
latent dims. If type='rna', z_dims==pooling_dims.
- --drop_prob
Drop_prob of TCR.
- --lr
Learning rate.
- --weight_decay
Weight_decay of learning rate.
- --max_epoch
Max epochs for training. 
- --patience
Max epochs for easy-stop.
- --warmup
Epochs for warm up.
- --gpu
Index of GPU to use if GPU is available. Default: 0.
- --seed
Random seed.
- --outdir
Output directory.

Citation
-------
**MIST**

Contacts
-------
kyzy850520@163.com
