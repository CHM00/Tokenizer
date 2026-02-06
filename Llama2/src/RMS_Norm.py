'''
    RMSNorm: Root Mean Square Layer Normalization
    通过均方根进行缩放，保留可学习的weight参数, 用于在归一化之后恢复模型的表达能力。
'''
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _normalize(self, x):
        '''torch.sqrt()计算平方根, torch.rsqrt()计算倒数平方根'''
        return x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

    def forward(self, x:torch.Tensor):
        x = self._normalize(x)
        x = x * self.weight
        return x

if __name__ == "__main__":
    # 准备参数
    batch_size = 4
    seq_len = 16
    dim = 512
    # torch.randn() 生成标准正态分布的随机数, np.random.randn() 生成标准正态分布的数组
    x = torch.randn(batch_size, seq_len, dim)
    print(x.shape)

    norm = RMSNorm(dim, 1e-5)
    out = norm(x)
    print("---RMSNorm Test---")
    print("x.shape: ", x.shape)
    print("out.shape: ", out.shape)

