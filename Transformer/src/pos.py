import torch
import torch.nn as nn
import math

'''
    正弦位置编码
    Transformer 论文中使用固定公式计算位置编码，不涉及可学习参数
'''
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dim = dim

        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, dim)

        # 生成位置索引 [0, 1, ..., max_seq_len-1] -> [max_seq_len, 1]
        position = torch.arrange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # 计算分母中的div_term: 10000^(2i/dim) = exp(2i * -log(10000)/dim)
        # 对数变换在数值计算上更稳定
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        # 填充PE矩阵
        # 偶数用sin, 奇数用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加batch维度: [max_seq_len, dim] -> [1, max_seq_len, dim]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # 注册为buffer，不作为模型参数更新

    def forward(self, x):
        # x.shape: [batch, seq_len, dim]
        seq_len = x.size(1)
        # 添加位置编码
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x