import torch
import torch.nn as nn
from sympy.physics.units.systems.si import dimex


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        self.dim = dim
        self.heads = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim * self.heads == dim, "dim must be divisible by n_heads"

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.q(q).view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(k).view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(v).view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(2, 3)) / self.dim ** 0.5

        if mask is not None:
            # mask == 0 的位置被填充为负无穷, Softmax 后变为 0
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax 与 加权求和
        atten_weights = torch.softmax(scores, dim=-1)

        if self.dropout is not None:
            atten_weights = self.dropout(atten_weights)
        out = torch.matmul(atten_weights, v)  # (batch_size, heads, seq_len, head_dim)

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    dim = 512
    n_heads = 8
    mha = MultiHeadAttention(dim, n_heads)
    x = torch.randn(batch_size, seq_len, dim)  # torch.randn表示生成一个服从标准正态分布的张量，np.random.randn表示生成一个服从标准正态分布的数组
    out = mha(x, x, x)
    print("---MultiHeadAttention Test---")
    print("x.shape: ", x.shape)
    print("out.shape: ", out.shape)