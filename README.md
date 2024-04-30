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
install the latest develop version
```bash
pip install git+https://github.com/aapupu/mist.git
```

or git clone and install
```bash
git clone git://github.com/aapupu/mist.git
cd mist
python setup.py install
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
- adata.h5ad
- model.pt

#### Option
- --rna_path
- --tcr_path
- --batch
- --rna_data_type
- --tcr_data_type
- --protein_path
- --type
- --min_genes
- --min_cells
- --pct_mt
- --n_top_genes
- --batch_size
- --pooling_dims
- --z_dims
- --drop_prob
- --lr
- --weight_decay
- --max_epoch
- --patience
- --gpu
- --seed
- --outdir

Citation
-------
**MIST**

Contacts
-------
kyzy850520@163.com
