'''
    自然语言处理中, 多对多的关系即序列到序列的学习 (Sequence to Sequence Learning, Seq2Seq, UnAligned) 是一种重要的任务,
    其目标是将一个输入序列转换为一个输出序列. 这种方法广泛应用于机器翻译、文本摘要、对话系统等领域.
'''
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
            bidirectional=False
        )

    def forward(self, x):
        # x.shape为(batch_size, seq_length)
        embedded = self.embedding(x)
        # 返回最终的隐藏状态和细胞状态作为上下文
        _, (hidden, cell) = self.lstm(embedded)  # 丢弃了序列中每一个时间步的隐藏状态output,只保留了最后一个时间步的隐藏状态和细胞状态
        return hidden, cell

# 解码器, 只是用了最后一步的隐藏状态和细胞状态做decoder的初始输入
class Decoder(nn.Module):
    '''
        解码器每一步接收一个词元和前一步的状态, 然后输出预测和新的状态。这个实现体现了为高效推理而设计的单步前向传播
    '''
    def __init__(self, vocab_size, hidden_size, nums_layer):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=nums_layer,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        # x.shape为(batch_size,)
        x = x.unsqueeze(1)  # (batch_size, 1)
        embedding = self.embedding(x)  # (batch_size, 1, hidden_size)

        # 接收上一步的状态, 计算当前步
        outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))
        print("outputs.shape: ",outputs.shape)
        predictions = self.fc(outputs.squeeze(1))
        print("predictions.shape: ",predictions.shape)
        return predictions, hidden, cell    # 返回的是当前步的预测

# 将上下文向量注入到输入中的解码器
class DecoderAlt(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(DecoderAlt, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        # 主要改动 1: RNN的输入维度是 词嵌入+上下文向量
        self.rnn = nn.LSTM(
            input_size=hidden_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hidden_ctx, hidden, cell):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)

        # 主要改动 2: 将上下文向量与当前输入拼接
        # 这里简单地取编码器最后一层的 hidden state 作为上下文代表
        context = hidden_ctx[-1].unsqueeze(1).repeat(1, embedded.shape[1], 1)
        rnn_input = torch.cat((embedded, context), dim=2)

        # 解码器的初始状态 hidden, cell 在第一步可设为零；之后需传递并更新上一步状态
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))
        return predictions, hidden, cell



# Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # 接收src的形状[batch, seq]
        # trg的形状[batch, trg]

        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        # 目标的词表大小
        trg_vocab_size = self.decoder.fc.out_features

        # 用于存储解码器的输出, 在每一个时间步的输出logits
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)  # hidden和cell的形状为(num_layers, batch_size, hidden_size)

        # 第一个输入是 <SOS>
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output

            # 决定是否使用教师强制
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            # 如果使用教师强制, 下一个输入是目标词元, 否则就是模型预测词元
            input = trg[:, t] if teacher_force else top1

        return outputs

    def greedy_decode(self, src, max_len=12, sos_idx=1, eos_idx=2):
        '''
            推理模式下的贪心解码
        '''
        self.eval()
        with torch.no_grad():
            hidden, cell = self.encoder(src)
            trg_indexes = [sos_idx]
            for _ in range(max_len):
                # 输入只有上一时刻的词元
                trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)

                # 解码一步, 并传入上一步的状态
                output, hidden, cell = self.decoder(trg_tensor, hidden, cell)

                # 获取当前步的预测, 并更新状态用于下一步
                pred_token = output.argmax(1).item()
                trg_indexes.append(pred_token)
                if pred_token == eos_idx:
                    break

        return trg_indexes

