'''
    Transformer的核心框架
    1、embedding
    2、Encoder层
    3、Decoder层
    4、Output层
'''

import torch.nn as nn
from .pos import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, hidden_dim, dim=512, n_heads=8, n_layers=6, max_seq_len=512, dropout=0.1):
        super(Transformer,self).__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.src_embedding = nn.Embedding(src_vocab_size, dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, dim)
        self.pos_encoder = PositionalEncoding(dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # 编码器与解码器堆叠
        # 使用ModuleList来存储层列表，支持按索引访问和自动注册参数
        self.encoder_layers = nn.ModuleList(
            EncoderLayer(dim, n_heads, hidden_dim, dropout) for _ in range(n_layers)
        )

        self.decoder_layers = nn.ModuleList(
            DecoderLayer(dim, n_heads, hidden_dim, dropout) for _ in range(n_layers)
        )

        self.output_layer = nn.Linear(dim, tgt_vocab_size)

    def forward(self, src, tgt):
        # 1、生成掩码（Padding Mask & Causal Mask）
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # 2、编码器前向传播
        enc_output = self.encoder(src, src_mask)

        # 3、解码器前向传播
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)

        # 4、输出logits
        logits = self.output_layer(dec_output)
        return logits
