'''
    Transformer的核心框架
    1、embedding
    2、Encoder层
    3、Decoder层
    4、Output层
'''
import torch
import torch.nn as nn
import math
# 导入组件
from .attention import MultiHeadAttention
from .ffn import FeedForward
from .norm import LayerNorm
from .pos import PositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # 多头自注意力机制
        self.self_attention = MultiHeadAttention(dim, n_heads, dropout)

        self.ffn = FeedForward(dim)
        self.layer_norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        attention = self.self_attention(src, src, src, src_mask)
        attention = self.dropout(attention)
        attention += src
        attention = self.layer_norm(attention)

        ffn_output = self.ffn(attention)
        ffn_output = self.dropout(ffn_output)
        ffn_output += attention
        out = self.layer_norm(ffn_output)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(dim, n_heads, dropout)
        self.self_norm = LayerNorm(dim)
        self.cross_attention = MultiHeadAttention(dim, n_heads, dropout)
        self.cross_norm = LayerNorm(dim)
        self.ffn = FeedForward(dim)
        self.ffn_norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        _x = tgt
        x = self.self_attention(_x, _x, _x, tgt_mask)
        x = self.self_norm(tgt + self.dropout(x))

        _x = x
        # 在 Cross-Attention 阶段，Decoder 的每一个词都在回头看 Encoder 输出的所有信息。
        _x = self.cross_attention(_x, enc_output, enc_output, src_mask)  # key 和 value 来自于 Encoder 的输出，即源句子的特征。
        x = self.cross_norm(x + self.dropout(_x))

        _x = x
        ffn_out = self.ffn(_x)
        ffn_out = self.ffn_norm(x + self.dropout(ffn_out))
        return ffn_out

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
            EncoderLayer(dim, n_heads, dropout) for _ in range(n_layers)
        )

        self.decoder_layers = nn.ModuleList(
            DecoderLayer(dim, n_heads, dropout) for _ in range(n_layers)
        )

        self.output_layer = nn.Linear(dim, tgt_vocab_size)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_mask(self, src, tgt):
        # src: batch_size, src_len
        # tgt: batch_size, tgt_len
        # 这里是padding mask
        src_mask = (src!=0).unsqueeze(1).unsqueeze(2)  # batch_size, 1, 1, src_len

        # padding mask, 训练中存在填充为0的样本
        tgt_len = tgt.size(1)
        tgt_mask = (tgt!=0).unsqueeze(1).unsqueeze(2)  # batch_size, 1, 1, tgt_len

        # padding mask 和 casual mask结合
        # 因果掩码和填充掩码结合
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        return src_mask, tgt_mask

    def encoder(self, src, src_mask):
        x = self.src_embedding(src) * math.sqrt(self.dim)  # 缩放嵌入, 当维度 dim 很大时（如 512 或 1024），每个维度的数值会因为分布原因变得相对较小。放大Embedding的量级
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decoder(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.tgt_embedding(tgt) * math.sqrt(self.dim)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x


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
