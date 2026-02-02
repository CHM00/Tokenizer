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
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # 计算分母中的div_term: 10000^(2i/dim) = exp(2i * -log(10000)/dim)
        # 对数变换在数值计算上更稳定
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))  # [dim//2], 广播机制

        # 填充PE矩阵
        # 偶数用sin, 奇数用cos, position * div_term 广播机制得到 [max_seq_len, dim//2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加batch维度: [max_seq_len, dim] -> [1, max_seq_len, dim]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # 注册为buffer，不作为模型参数更新

    def forward(self, x):
        # x.shape: [batch, seq_len, dim]  将位置编码加到输入的词嵌入上
        seq_len = x.size(1)
        # 添加位置编码
        # 截取与输入序列长度对应的位置编码并相加
        # x.size(1) 是 seq_len
        # self.pe 的形状是 [1, max_seq_len, dim]，切片后会自动广播到 batch_size
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x

if __name__ == "__main__":
    batch = 2
    seq_len = 10
    dim = 512
    max_seq_len = 500
    # 初始化模块
    pe = PositionalEncoding(dim, max_seq_len)

    # 准备输入
    x = torch.zeros(batch, seq_len, dim) # 输入为0, 直接观察PE值

    # 前向传播
    output = pe(x)

    # 验证输出
    print("---PositionalEncoding Test---")
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)