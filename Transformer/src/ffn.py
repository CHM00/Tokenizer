import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super(FeedForward, self).__init__()
        self.fn1 = nn.Linear(dim, dim * 4)
        self.fn2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fn2(self.dropout(torch.relu(self.fn1(x))))

if __name__ == "__main__":
    # 准备参数
    batch_size = 2
    seq_len = 10
    dim = 512

    ffn = FeedForward(dim, dropout=0.2)
    x = torch.randn(batch_size, seq_len, dim)
    out = ffn(x)
    print("---FFN Test---")
    print("x.shape: ", x.shape)
    print("out.shape: ", out.shape)