import torch
import torch.nn as nn


# 实现1，问题在于embedding层是共享的, 不应该放到self-Attention里面
class SelfAttention(nn.Module):
    def __init__(self, seq_len, embedding_size):
        super(SelfAttention, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(seq_len, embedding_size)
        self.q = nn.Linear(embedding_size, embedding_size)
        self.k = nn.Linear(embedding_size, embedding_size)
        self.v = nn.Linear(embedding_size, embedding_size)


    def _softmax(self, x):
        exp_x = torch.exp(x)
        return  exp_x / torch.sum(exp_x, dim=-1, keepdim=True)

    def forward(self,x):
        # x.shape是[batch, seq_len]
        batch_size, seq_len = x.shape
        embedding = self.embedding(x)
        # batch, seq, embedding_size
        q = self.q(embedding)
        k = self.k(embedding)
        v = self.v(embedding)

        score = torch.matmul(q, k.transpose(1,2)) / torch.sqrt(self.embedding_size)
        attention_score = self._softmax(score)  # batch, seq, seq

        new_embedding = torch.matmul(attention_score, v)

        return new_embedding

# 多头自注意力机制
class Multi_Self_Attention(nn.Module):
    def __init__(self, embedding_size, head):
        super(Multi_Self_Attention, self).__init__()
        self.head = head
        self.embedding_size = embedding_size
        self.fq = nn.Linear(embedding_size, embedding_size)
        self.fk = nn.Linear(embedding_size, embedding_size)
        self.fv = nn.Linear(embedding_size, embedding_size)
        self.head_dim = embedding_size // head
        assert self.head_dim * self.head == embedding_size
        self.fc = self.Linear(embedding_size, embedding_size)

    def _softmax(self, x):
        exp_x = torch.exp(x)
        return exp_x / torch.sum(exp_x, dim=-1, keepdim=True) + 1e-8

    def forward(self, x):
        batch, seq, embed = x.shape
        q = self.fq(x).view(batch, seq, self.head, self.head_dim).permute(0, 2, 1, 3)
        k = self.fk(x).view(batch, seq, self.head, self.head_dim).permute(0, 2, 1, 3)
        v = self.fv(x).view(batch, seq, self.head, self.head_dim).permute(0, 2, 1, 3)

        score = torch.matmul(q, k.transpose(2, 3)) / self.head_dim ** 0.5
        attention_score = self._softmax(score) # torch.softmax(score, dim=-1)

        new_embedding = torch.matmul(attention_score, v)
        new_embedding = new_embedding.permute(0, 2, 1, 3).contiguous().view(batch, seq, -1)
        new_embedding = self.fd(new_embedding)  #虽然各个头是并行计算的, 但是最后还是要把各个头的结果拼接起来，然后通过一个线性层进行变换
        return new_embedding