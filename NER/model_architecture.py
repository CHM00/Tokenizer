'''
    在pytorch中,nn.ModuleList和nn.Sequential都是用来容纳多个子模块的容器, 但它们的设计思想和使用场景不同:
    nn.Sequential: 像一个自动化的流水线, 数据会自动按顺序流过每一层, 适用于简单的线性堆叠, 但无法实现层间的复杂交互
    nn.ModuleList: 像一个普通的Python列表, 只负责存储模块, 而不会自动执行它们。 你需要在forward方法中手动编写循环调用每一层, 可以在层与层之间加入自定义逻辑(如残差连接)
    词向量的维度和GRU的隐状态维度hidden_size保持一致
'''
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.utils.rnn as rnn

class GRUNerNetWork(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_tags, num_gru_layers=1):
        super(GRUNerNetWork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # 使用 ModuleList 来存储多层 GRU
        self.gru_layers = nn.ModuleList()
        for _ in range(num_gru_layers):
            self.gru_layers.append(
                nn.GRU(input_size=hidden_size,
                       hidden_size=hidden_size,
                       num_layers=1,
                       batch_first=True,
                       bidirectional=False)
            )

        # 分类决策层
        self.fc = nn.Linear(hidden_size, num_tags)

    def forward(self, token_ids, attention_mask=None):
        # [batch_size, seq_length] -> [batch_size, seq_length, hidden_size]
        embedded_text = self.embedding(token_ids)

        current_input = embedded_text
        for gru_layer in self.gru_layers:
            # GRU的输出: output, h_n
            output, _ = gru_layer(current_input)
            current_input = output + embedded_text  # 残差连接

        logits = self.fc(current_input)

        return logits  # [batch_size, seq_length, num_tags]




class BiGRUNerNetWork(nn.Module):
    '''
        双向 GRU 包含一个从右到左的反向传播路径。它会从序列的末尾开始计算，如果末尾都是无意义的 <PAD> 标记，
        那么这些“垃圾信息”就会作为初始状态，一路污染到序列中真实的 Token 表示中去。所以，需要一种方法来“告知”GRU 每个序列的真实长度，
        让它在计算时能够忽略掉这些填充位。
    '''

    def __init__(self, vocab_size, hidden_size, num_tags, num_gru_layers=1):
        super(BiGRUNerNetWork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # 使用 ModuleList 来存储多层 GRU
        self.gru_layers = nn.ModuleList()
        for _ in range(num_gru_layers):
            self.gru_layers.append(
                nn.GRU(input_size=hidden_size,
                       hidden_size=hidden_size,
                       num_layers=1,
                       batch_first=True,
                       bidirectional=True)
            )

        # 3. 特征融合层
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        # 4、分类决策层
        self.classifier = nn.Linear(hidden_size, num_tags)

    def forward(self, token_ids, attention_mask=None):
        # 1. 计算真实长度
        lengths = attention_mask.sum(dim=1).cpu()

        # 2. 获取词向量
        # [batch_size, seq_length] -> [batch_size, seq_length, hidden_size]
        embedded_text = self.embedding(token_ids)

        # 3. 打包序列
        current_packed_input = rnn.pack_padded_sequence(
            embedded_text, lengths, batch_first=True, enforce_sorted=False
        )

        # current_input = embedded_text
        for gru_layer in self.gru_layers:
            # GRU的输出: output, h_n
            packed_output, _ = gru_layer(current_packed_input)

            # 解包以进行后续操作，并指定 total_length
            output, _ = rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=token_ids.shape[1]
            )

            # 特征融合
            features = self.fc(output)

            # 残差连接
            # 同样需要解包上一层的输入
            input_padded, _ = rnn.pad_packed_sequence(
                current_packed_input, batch_first=True, total_length=token_ids.shape[1]
            )
            current_input = features + input_padded

            # 重新打包作为下一层的输入
            current_packed_input = rnn.pack_padded_sequence(
                current_input, lengths, batch_first=True, enforce_sorted=False
            )

        # 5. 解包最终输出用于分类
        final_output, _ = rnn.pad_packed_sequence(
            current_packed_input, batch_first=True, total_length=token_ids.shape[1]
        )

        # 6. 分类
        logits = self.classifier(final_output)

        return logits  # [batch_size, seq_length, num_tags]






if __name__ == "__main__":
    # 测试模型
    token_ids = torch.tensor([
        [210, 18, 871, 147, 0, 0, 0, 0],
        [922, 2962, 842, 210, 18, 871, 147, 0]
    ], dtype=torch.int64)

    # attention_mask 标记哪些是真实 token (1) 哪些是填充 (0)
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0]
    ], dtype=torch.int64)

    label_ids = torch.tensor([
        [0, 0, 0, 0, -100, -100, -100, -100],
        [0, 0, 0, 0, 0, 0, 0, -100]
    ], dtype=torch.int64)


    model = GRUNerNetWork(
        vocab_size=10000,
        hidden_size=128,
        num_tags=37,
        num_gru_layers=2
    )

    # 执行前向传播
    logits = model(token_ids, attention_mask)

    # 构造损失函数
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # reduction="none"表示损失张量的形状为[2,8]与label_ids相同, 返回了每个Token各自的损失。

    # 计算损失
    # CrossEntropyLoss 要求维度在前, 所以需要交换最后两个维度
    # batch, seq_len, num_tags -> batch, num_tags, seq_len  交换两个维度
    permuted_logits = torch.permute(logits, (0, 2, 1))
    loss = loss_fn(permuted_logits, label_ids)

    print(f"Logits shape: {logits.shape}")  # 应该是 [2, 8, 37]
    print(f"Loss shape: {loss.shape}")      # 应该是 [2, 8]
    print("\n每个token的损失:")
    print(loss)  # 每个 token 的损失
