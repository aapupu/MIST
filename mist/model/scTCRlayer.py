import torch
from torch.distributions import Normal
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0,  d_model, step=2).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        _, seq_len = x.size()
        return self.encoding[:seq_len, :].unsqueeze(0).to(x.device) #[1, seq_len, d_model]

class AminoAcidEmbedding(nn.Module):
    def __init__(self, aa_size=21, aa_dims=64, pad_token_id=0, 
                 max_len=30, drop_prob=0.1):
        super().__init__()
        self.embedding = nn.Embedding(aa_size, aa_dims, padding_idx=pad_token_id)
        self.pos_emb = PositionalEncoding(aa_dims, max_len)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        return self.dropout(self.pos_emb(x)+self.embedding(x))

class ConvBlock(nn.Module):
    def __init__(self, config):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(**config['conv_params'], bias=False)
        self.bn = nn.BatchNorm2d(config['conv_params']['out_channels'])
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class SelfOutput(nn.Module):
    def __init__(self, dims=64, drop_prob=0.1):
        super(SelfOutput, self).__init__()
        self.linear = nn.Linear(dims, dims)
        self.LayerNorm = nn.LayerNorm(dims, eps=1e-12)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, input_x):
        x = self.dropout(self.linear(x))
        x = self.LayerNorm(input_x + x)
        return x
    
class CDR3Extractor(nn.Module):
    def __init__(self, dims=64, drop_prob=0.1, conv_config=None):
        super().__init__()
        self.self = nn.MultiheadAttention(dims, num_heads=4, 
                                          dropout=drop_prob, 
                                          batch_first=True)
        self.self_output= SelfOutput(dims, drop_prob)
        self.conv_blocks = nn.ModuleList([
            ConvBlock(config) for config in conv_config
        ])
        # self.pooling = nn.AdaptiveAvgPool2d((15,1))
        self.flatten = nn.Flatten()
        
    def _forward_conv(self, x):
        x = x.unsqueeze(1)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        return x
        
    def forward(self, x, attention_mask):
        self_outputs = self.self(x,x,x,
                    attn_mask=None, 
                    key_padding_mask=attention_mask)
        x = self.self_output(self_outputs[0], x) 
        x = self._forward_conv(x)
        x = self.flatten(x)
        return x

class GeneEncoder(nn.Module):
    def __init__(self, gene_size=50, gene_dims=48, pad_token_id=0):
        super().__init__()
        self.embedding = nn.Embedding(gene_size, gene_dims, padding_idx=pad_token_id)
        
    def forward(self, x):
        x = self.embedding(x)
        return x
    
class encoder_scTCR(nn.Module):
    def __init__(self, z_dims=32, 
                 aa_size=21, aa_dims=64, max_len=30, 
                 bv_size=None, bj_size=None,
                 av_size=None, aj_size=None, 
                 gene_dims=48,
                 drop_prob=0.1, 
                 conv_config=None):
        super().__init__()
        self.aa_embedding = AminoAcidEmbedding(aa_size=aa_size, aa_dims=aa_dims, max_len=max_len, drop_prob=drop_prob)
        self.cdr3b_encode = CDR3Extractor(dims=aa_dims, drop_prob=drop_prob, conv_config=conv_config)
        self.cdr3a_encode = CDR3Extractor(dims=aa_dims, drop_prob=drop_prob, conv_config=conv_config)
        
        self.bv_encode =  GeneEncoder(gene_size=bv_size, gene_dims=gene_dims)
        self.bj_encode =  GeneEncoder(gene_size=bj_size, gene_dims=gene_dims)
        
        self.av_encode =  GeneEncoder(gene_size=av_size, gene_dims=gene_dims)
        self.aj_encode =  GeneEncoder(gene_size=aj_size, gene_dims=gene_dims)
        
        self.fc = nn.Linear(3776, 512) #2112 3776
        self.fc_mu = nn.Linear(512, z_dims)
        self.fc_var = nn.Linear(512, z_dims)
        
        self.bn = nn.BatchNorm1d(512)
        self.act = nn.ReLU()
        
    def reparameterize(self, mu, var):
        z = Normal(mu, var.sqrt()).rsample()
        return z

    def forward(self, bv=None, bj=None, cdr3b=None, 
                av=None, aj=None, cdr3a=None):
        attn_mask_b = (cdr3b != 0).clone().detach()
        attn_mask_a = (cdr3a != 0).clone().detach()
        cdr3b, cdr3a = self.aa_embedding(cdr3b), self.aa_embedding(cdr3a)
        cdr3b, bv, bj = self.cdr3b_encode(cdr3b, attn_mask_b), self.bv_encode(bv), self.bj_encode(bj)
        cdr3a, av, aj = self.cdr3a_encode(cdr3a, attn_mask_a), self.av_encode(av), self.aj_encode(aj)
        
        h = torch.cat([cdr3b, bv, bj, cdr3a, av, aj], dim=-1)
        h = self.act(self.bn(self.fc(h)))
        mu = self.fc_mu(h)
        var = torch.exp(self.fc_var(h))
        z = self.reparameterize(mu, var)
        return z, mu, var

class ConvTransposeBlock(nn.Module):
    def __init__(self, config):
        super(ConvTransposeBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(**config['conv_transpose_params'], bias=False)
        self.bn = nn.BatchNorm2d(config['conv_transpose_params']['out_channels'])
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, drop_prob=0.1):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
    
class GeneDecoder(nn.Module):
    def __init__(self, z_dims=32, gene_size=50,  gene_dims=48, drop_prob=0.1, embedding_weight=None):
        super(GeneDecoder, self).__init__()
        self.linear_block = LinearBlock(z_dims, gene_dims, drop_prob)
        self.decoder = nn.Linear(gene_dims, gene_size, bias=False)            
        if embedding_weight is not None:
            self.decoder.weight = nn.Parameter(embedding_weight)  
        
    def forward(self, x):
        return self.decoder(self.linear_block(x))
    
class CDR3Decoder(nn.Module):
    def __init__(self, z_dims=32, aa_size=21, aa_dims=64, max_len=30, drop_prob=0.1, 
                 embedding_weight=None, conv_transpose_configs=None):
        super().__init__()
        self.linear_block1 = LinearBlock(z_dims, 256, drop_prob)
        self.conv_transpose_blocks = nn.ModuleList([
            ConvTransposeBlock(config) for config in conv_transpose_configs
        ])
        self.linear_block2 = LinearBlock(aa_dims, aa_dims, drop_prob)
        
        self.decoder = nn.Linear(aa_dims, aa_size, bias=False)           
        if embedding_weight is not None:
            self.decoder.weight = nn.Parameter(embedding_weight)  
        
        self.max_len = max_len
        self.aa_dims = aa_dims
        
    def forward(self, x):
        x = self.linear_block1(x)
        x = x.view(-1, 16, 4, 4)
        for conv_transpose_block in self.conv_transpose_blocks:
            x = conv_transpose_block(x)
        x = x.view(-1, self.max_len, self.aa_dims)
        x = self.linear_block2(x)
        return self.decoder(x)
    
class decoder_scTCR(nn.Module):
    def __init__(self, z_dims=32, aa_size=21, aa_dims=64, max_len=30, 
                 bv_size=None, bj_size=None,
                 av_size=None, aj_size=None, 
                 gene_dims=48,
                 drop_prob=0.1,
                 aa_embedding_weight=None,
                 bv_embedding_weight=None,bj_embedding_weight=None,
                 av_embedding_weight=None,aj_embedding_weight=None,
                 conv_transpose_configs=None):
        super().__init__()
        self.CDR3b_decode =  CDR3Decoder(z_dims, aa_size, aa_dims, max_len, drop_prob, 
                                         aa_embedding_weight, conv_transpose_configs)
        self.bv_decode = GeneDecoder(z_dims, bv_size,  gene_dims, drop_prob, bv_embedding_weight)
        self.bj_decode = GeneDecoder(z_dims, bj_size,  gene_dims, drop_prob, bj_embedding_weight)
        
        self.CDR3a_decode =  CDR3Decoder(z_dims, aa_size, aa_dims, max_len, drop_prob, 
                                         aa_embedding_weight, conv_transpose_configs)
        self.av_decode = GeneDecoder(z_dims, av_size,  gene_dims, drop_prob, av_embedding_weight)
        self.aj_decode = GeneDecoder(z_dims, aj_size,  gene_dims, drop_prob, aj_embedding_weight)

    def forward(self, x):
        recon_bv, recon_bj, recon_cdr3b = self.bv_decode(x), self.bj_decode(x), self.CDR3b_decode(x)
        recon_av, recon_aj, recon_cdr3a = self.av_decode(x), self.aj_decode(x), self.CDR3a_decode(x)
        return recon_bv, recon_bj, recon_cdr3b, recon_av, recon_aj, recon_cdr3a
