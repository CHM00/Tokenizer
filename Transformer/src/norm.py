import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    '''
        LayerNorm层实现
        公式: y = (x - mean) / sqrt(var + eps) * gamma + beta
    '''
    def __init__(self, dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        # 可学习参数gamma和beta
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x.shape: (batch_size, seq_len, dim)

        # 计算最后一维特征维度的均值和方差
        mean = x.mean(dim=-1, keepdim=True)  # 计算均值

        # 计算的是有偏方差
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算方差

        # 归一化
        x_normlized = (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta

        return x_normlized

if __name__ == "__main__":
    # 准备参数
    batch_size = 2
    seq_len = 10
    dim = 512

    layernorm = LayerNorm(dim, eps=1e-6)
    x = torch.randn(batch_size, seq_len, dim)
    out = layernorm(x)
    print("---LayerNorm Test---")
    print("x.shape: ", x.shape)
    print("out.shape: ", out.shape)