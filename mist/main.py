#!/usr/bin/env python
import argparse

def main():
    parser = argparse.ArgumentParser(description='MIST: Deep Learning Integration of Single-Cell RNA and TCR Sequencing Data for T-Cell Insight')
    
    parser.add_argument('--rna_path', '-r', type=str, nargs='*', default=None)
    parser.add_argument('--tcr_path', '-t', type=str, nargs='*', default=None)
    parser.add_argument('--protein_path', '-p', type=str, nargs='?', default=None)
    parser.add_argument('--batch', '-b', type=str, nargs='*', default=None)
    parser.add_argument('--rna_data_type', '-rt', type=str, nargs='?', default='h5ad')
    parser.add_argument('--tcr_data_type','-tt', type=str, nargs='?', default='10X')
    parser.add_argument('--type', type=str, nargs=1, default='multi')
    
    parser.add_argument('--min_genes', type=int, default=600)
    parser.add_argument('--min_cells', type=int, default=3)
    parser.add_argument('--pct_mt', type=int, default=None)
    parser.add_argument('--n_top_features', default=2000)
    parser.add_argument('--backed', action='store_true', default=False)
    parser.add_argument('--batch_scale', action='store_true', default=False)
    parser.add_argument('--batch_min', type=int, default=0)
    parser.add_argument('--remove_TCRGene', action='store_true', default=False)
    
    # parser.add_argument('--max_len', type=int, default=30)
    # parser.add_argument('--scirpy', action='store_true', default=True)
    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pooling_dims', type=int, default=16)
    parser.add_argument('--z_dims', type=int, default=128)
    parser.add_argument('--drop_prob', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--warmup', type=int, default=40)
    parser.add_argument('--penalty', type=str, default='mmd_rbf')
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', '-o', type=str, default='mist_output/')

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

if __name__ == '__main__':
    main()
        
