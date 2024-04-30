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
adata, model = MIST()
```

### 2. Command line
```bash
MIST --
```

#### Output 
- adata.h5ad
- model.pt

#### Option


Citation
-------
**MIST**


Contacts
-------
kyzy850520@163.com
