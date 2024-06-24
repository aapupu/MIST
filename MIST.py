#!/usr/bin/env python
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIST: an interpretable and flexible deep learning framework for single-T cell transcriptome and receptor analysis')
    
    parser.add_argument('--rna_path', '-r', type=str, nargs='*', default=None, help='Paths to scRNA-seq data files')
    parser.add_argument('--tcr_path', '-t', type=str, nargs='*', default=None, help='Paths to scTCR-seq data files')
    parser.add_argument('--protein_path', '-p', type=str, nargs='?', default=None, help='Path to merged protein (ADT) data file')
    parser.add_argument('--batch', '-b', type=str, nargs='*', default=None, help='Batch labels')
    parser.add_argument('--rna_data_type', '-rt', type=str, nargs='?', default='h5ad', help='Type of scRNA-seq data file (e.g., 10X mtx, h5, or h5ad). Default: h5ad')
    parser.add_argument('--tcr_data_type','-tt', type=str, nargs='?', default='10X', help='Type of scTCR-seq data file (e.g., 10X, tracer, BD, or h5ad). Default: 10X')
    parser.add_argument('--type', type=str, nargs=1, default='joint', help='Type of model to train (joint, rna, or tcr). Default: joint')
    
    parser.add_argument('--min_genes', type=int, default=600, help='Filtered out cells that are detected in less than min_genes. Default: 600')
    parser.add_argument('--min_cells', type=int, default=3, help='Filtered out genes that are detected in less than min_cells. Default: 3')
    parser.add_argument('--pct_mt', type=int, default=None, help='Filtered out cells that are detected in more than percentage of mitochondrial genes. If None, Filtered out mitochondrial genes. Default: None')
    parser.add_argument('--n_top_features', default=2000, help='Number of highly-variable genes to keep. Default: 2000')
    parser.add_argument('--backed', action='store_true', default=False, help='Whether to use backed format for reading data. Default: False')
    parser.add_argument('--batch_scale', action='store_true', default=False, help='Whether to data scale pre batch. Default: False')
    parser.add_argument('--batch_min', type=int, default=0, help='Filtered out batch that are detected in less than cells. Default: 0')
    parser.add_argument('--remove_TCRGene', action='store_true', default=False, help='Whether to remove TCR gene (e.g., TRAV1-2). Default: False')
    
    # parser.add_argument('--max_len', type=int, default=30)
    # parser.add_argument('--scirpy', action='store_true', default=True)
    
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training. Default: 128')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading. Default: 8')
    parser.add_argument('--pooling_dims', type=int, default=16, help='Dimensionality of pooling layer. Default: 16')
    parser.add_argument('--z_dims', type=int, default=128, help='Dimensionality of latent space. Default: 128')
    parser.add_argument('--drop_prob', type=float, default=0.1, help=' Dropout probability. Default: 0.1')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate. Default: 1e-4')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help=' Weight decay. Default: 1e-3')
    parser.add_argument('--max_epoch', type=int, default=400, help='Maximum number of epochs. Default: 400')
    parser.add_argument('--patience', type=int, default=40, help='Patience for early stopping. Default: 40')
    parser.add_argument('--warmup', type=int, default=40, help='Warmup epochs. Default: 40')
    parser.add_argument('--penalty', type=str, default='mmd_rbf', help='Type of penalty loss. Default: mmd_rbf')
    
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU to use if GPU is available. Default: 0')
    parser.add_argument('--seed', type=int, default=42, help='Random seed. Default: 42')
    parser.add_argument('--outdir', '-o', type=str, default='mist_output/', help='Output directory')

    # parser.add_argument('--aa_dims', type=int, default=64)
    # parser.add_argument('--gene_dims', type=int, default=48)
    # parser.add_argument('--weights', action='store_true', default=False)
    
    args = parser.parse_args()
    
    from mist import MIST
    adata, model = MIST(
        rna_path=args.rna_path, 
        tcr_path=args.tcr_path, 
        batch=args.batch,
        rna_data_type=args.rna_data_type,
        tcr_data_type=args.tcr_data_type,
        protein_path=args.protein_path,
        type=args.type,         
        min_genes=args.min_genes, 
        min_cells=args.min_cells, 
        pct_mt=args.pct_mt,
        n_top_genes=args.n_top_genes, 
        backed=args.backed,
        batch_scale=args.batch_scale,
        batch_min=args.batch_min,
        remove_TCRGene=args.remove_TCRGene,
        max_len=30,
        scirpy=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aa_dims=64,
        gene_dims=48,
        pooling_dims=args.pooling_dims,
        z_dims=args.z_dims,
        drop_prob=args.drop_prob,
        weights=False,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epoch=args.max_epoch,
        patience=args.patience, 
        warmup=args.warmup,
        penalty=args.penalty,
        gpu=args.gpu, 
        seed=args.seed,
        outdir=args.outdir
    )
        
