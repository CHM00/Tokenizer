import torch
import torch.nn as nn
import random
# 标准的Encoder-Decoder架构

# 编码器
class Encoder(nn.Module):
    '''
        将单向LSTM的隐藏状态hidden和细胞状态cell作为上下文传递给解码器
    '''
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x.shape为(batch_size, seq_length)
        embedded = self.embedding(x)
        # 返回最终的隐藏状态和细胞状态作为上下文
        outputs, (hidden, cell) = self.lstm(embedded)
        # 将双向RNN的输出通过线性层降维，使其与解码器维度匹配
        outputs = torch.tanh(self.fc(outputs))  # (batch, src_len, hidden_size * 2)

        return outputs, hidden, cell


def _softmax(x):
    expx = torch.exp(x)
    # 为什么沿着最后一维求和, 因为最后一维通常表示不同类别的分数或权重（表示不同时间步的隐藏状态和前一步隐藏状态的相似度）, 然后对最后一维进行归一化
    return expx / torch.sum(expx, -1, keepdim=True)

class AttentionSample(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionSample, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs):
        # 隐藏层的形状: (num_layers, batch_size, hidden_size)
        # outputs形状: (batch_size, seq_length, hidden_size)
        query = hidden[-1]  # 取最后一层的隐藏状态作为查询 (batch_size, hidden_size)
        query = query.unsqueeze(1) # (batch_size, 1, hidden_size)
        key = encoder_outputs
        value = encoder_outputs # (batch_size, seq_length, hidden_size)
        score = torch.matmul(query, key.transpose(1,2)) / torch.sqrt(self.hidden_size)  # batch_size, 1, seq_length
        score = _softmax(score)  # batch_size, 1, seq_length
        return score.squeeze(1)  # (batch_size, seq_length)



# def attention(hidden, encoder_outputs):
#     # 隐藏层的形状: (num_layers, batch_size, hidden_size)
#     # outputs形状: (batch_size, seq_length, hidden_size)



# 解码器, 只是用了最后一步的隐藏状态和细胞状态做decoder的初始输入
class Decoder(nn.Module):
    '''
        解码器每一步接收一个词元和前一步的状态, 然后输出预测和新的状态。这个实现体现了为高效推理而设计的单步前向传播
    '''
    def __init__(self, vocab_size, hidden_size, nums_layer, attention):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention = attention
        self.lstm = nn.LSTM(
            input_size=hidden_size * 2,   # 注意力机制将上下文向量和输入向量拼接在一起，作为LSTM的输入
            hidden_size=hidden_size,
            num_layers=nums_layer,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        # x.shape为(batch_size,)
        x = x.unsqueeze(1)  # (batch_size, 1)
        embedding = self.embedding(x)  # (batch_size, 1, hidden_size)

        # 计算注意力分数
        att = self.attention(hidden, encoder_outputs).unsqueeze(1) # (batch_size, 1, seq_length)

        # 计算上下文向量
        context = torch.matmul(att, encoder_outputs)  # batch_size, 1, hidden_size

        # 将上下文向量与嵌入向量拼接
        embedding = torch.cat((embedding, context), dim=2)  # (batch_size

        # 接收上一步的状态, 计算当前步
        outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))
        print("outputs.shape: ",outputs.shape)
        predictions = self.fc(outputs.squeeze(1))
        print("predictions.shape: ",predictions.shape)
        return predictions, hidden, cell    # 返回的是当前步的预测