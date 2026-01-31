from sklearn.datasets import fetch_20newsgroups
from torch import no_grad

# 选取四个类别的数据
col = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
train = fetch_20newsgroups(subset='train',categories=col,shuffle=True, random_state=42)
test = fetch_20newsgroups(subset='test',categories=col,shuffle=True, random_state=42)

sample = {
    "text_preview": train.data[0][:200],
    "label": train.target_names[train.target[0]]
}

print("sample: ", sample)

import matplotlib.pyplot as plt
import re

# 为了进行探索，先定义一个简单的分词函数
def basic_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9(),.!?\\'`]", " ", text)  # 目的是为了排除不在[]中的所有字符, 排除中文, 制表符等
    text = re.sub(r"([,.!?\\'`])", r" \\1 ", text)  # 在标点符号前后加空格
    tokens = text.strip().split()  # 按空格分词
    return tokens

# 计算每篇文档的词元数量
train_text_lengths = [len(basic_tokenize(text)) for text in train.data]

plt.figure(figsize=(10, 6))
plt.hist(train_text_lengths, bins=50, alpha=0.7, color='blue')  # bins=50表示将数据分成了50个区间，每个区间都是一个长度组
plt.title('Distribution of Text Lengths in Training Data')
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 这里图片中展示的是, 训练集中不同token数量的文档有多少篇, 各自出现了多少次, 分析训练集中文档长度的分布规律，为后续NLP处理做准备。

from collections import Counter
import numpy as np

# 计算所有词元的频率
word_count = Counter()
for text in train.data:
    # 计算得到所有的Token值
    word_count.update(basic_tokenize(text))

# 获取频率并按照降序排序
frequencies = sorted(word_count.values(), reverse=True)

# 生成排名
rank = np.arange(1, len(frequencies)+1)

# 绘制对数坐标图
plt.figure(figsize=(10, 6))
plt.loglog(rank, frequencies, color='blue')
plt.title("Word Frequency Distribution (Log-Log Scale)")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# 形成一个Tokenizer的类, 用于分词、词典构建以及ID转换的任务
'''
    分词策略: 基于正则表达式的分词策略，先将文本转为小写，然后通过re.sub移除非字母、数字和基本标点之外的字符，接着在标点符号前后添加空格，最后按空格分词。
    词典构建: 使用训练数据中的所有词元，统计其频率，过滤掉出现次数较少的低频词，以减少词典规模和噪声。同时词典初始化时会预设两个特殊的Token:"<PAD>"和"<UNK>"，分别用于填充和表示未知词元。
'''
class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token_to_id = { token:idx for token, idx in self.vocab.items() }


    @classmethod
    def _tokenizer_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9(),.!?\\'`]", " ", text)  # 目的是为了排除不在[]中的所有字符, 排除中文, 制表符等
        text = re.sub(r"([,.!?\\'`])", r" \\1 ", text)  # 在标点符号前后加空格
        tokens = text.strip().split()  # 按空格分词
        return tokens

    def tokenize(self, text):
        return self._tokenizer_text(text)

    # 转换token成id
    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_id.get(token, self.vocab["<UNK>"]) for token in tokens]

    def __len__(self):
        return len(self.vocab)

# 词典只包含在训练集中出现超过min_freq次的词元
def build_vocab(word_count, min_freq=5):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in word_count.items():
        if count >= min_freq:
            vocab[word] = len(vocab)  # 用于表示词的索引
    return vocab

vocab = build_vocab(word_count, min_freq=5)
tokenizer = Tokenizer(vocab)
print("Vocabulary Size: ", len(tokenizer))


# 如何处理超长token, 直接处理过长token会导致内存溢出和计算效率低下
'''
    解决办法: 将一篇长文档切分成多个固定长度、且有部分重叠的文本块，每个文本块的长度为max_length，重叠部分的长度为overlap_size。
    好处: ① 信息保全: 完整的利用了整篇文章的信息 ② 数据增强: 将一篇长文档变成了多条训练样本, 增加了训练量，有助于模型学习 
'''

# 封装Dataset和DataLoader
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

class TextDataset(Dataset):
    '''
        接受原始文本，调用tokenzier进行ID化，并生成适合模型输入的张量数据，应用滑动分割策略处理长文本。如果文本超过max_len, 则会进行切分。
    '''
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.processed_data=[]

        for text, label in tqdm(zip(texts, labels), total=len(labels)):
            # 文本转换为token, 进行ID化
            token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

            if len(token_ids) > self.max_len:
                stride = max(1, int(self.max_len * 0.8))
                for i in range(0, len(token_ids)- self.max_len + 1, stride):
                    chunk = token_ids[i:i+self.max_len]
                    self.processed_data.append({"token_ids": chunk, "label": label})
            else:
                self.processed_data.append({"token_ids": token_ids, "label": label})

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


def collate_fn(batch):
    '''
        负责处理一个批次内长短不一的样本，通过填充统一长度
    '''
    max_batch_len = max(len(item["token_ids"]) for item in batch)

    batch_token_ids, batch_labels = [], []

    for item in batch:
        token_ids = item["token_ids"]
        padding_len = max_batch_len - len(token_ids)
        padded_ids = token_ids + ([0] * padding_len)
        batch_token_ids.append(padded_ids)
        batch_labels.append(item["label"])

    return {
        "token_ids": torch.tensor(batch_token_ids, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long)
    }

from torch.utils.data import DataLoader

train_dataset = TextDataset(train.data, train.target, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

valid_dataset = TextDataset(test.data, test.target, tokenizer)
valid_loader = DataLoader(valid_dataset, batch_size=32, collate_fn=collate_fn)

print({"train_samples": len(train_dataset), "valid_samples": len(valid_dataset), "batch_size": 32})


## 模型构建，梳理清楚数据的变形记，即张量如何在网络变化
'''
    Input:
        token_ids(词元ID序列): [batch_size, seq_len]
                    |
    Embedding:
        nn.Embedding(padding_idx=0)
                    |
    embedded: [batch_size, seq_len, embed_dim]
                    |
    Linear:
        nn.Linear(embed_dim, hidden_dim * 2)
                    |
                nn.ReLU
                    |
        nn.Linear(hidden_dim * 2, hidden_dim * 4)
                    |
                nn.ReLU
                    |
    token_features:[batch_size, seq_len, embed_dim * 4]
                    |
            Masked Average Pooling (掩码平均池化, 关键操作)
                    |
    pool_features:[batch_size, hidden_dim * 2] (聚合seq_len维度)
                    |
            nn.Linear(分类层)
                    |
    Output:
        logits:[batch_size, num_classed]
'''

# 掩码平均池化
'''
    池化的目的是将一个序列的特征 ([seq_len, hidden_dim]) 聚合成一个代表整条序列的向量 ([hidden_dim]), 简单的平均池化会受到填充<PAD>的影响, 导致语义偏差
'''

# 模型代码
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(TextClassifier, self).__init__()
        # 可学习的矩阵
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # 如果看到 ID 为 0 的位置，那是为了补齐长度凑数的，不需要计算梯度。

        self.feature_extractor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU()
        )

        self.classifier = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        token_feature= self.feature_extractor(embedded)  # token_feature: [batch_size, seq_len, hidden_dim * 4]

        # ---掩码平均池化---
        # 判断 token_ids 中哪些不是 0（即不是 padding）。
        # padding_mask 会变成一个由 1.0 和 0.0 组成的矩阵。
        padding_mask = (token_ids != self.embedding.padding_idx).float() # [batch_size, seq_len]
        # padding_mask.unsqueeze(-1) 将形状从 [batch_size, seq_len] 变成 [batch_size, seq_len, 1]
        # 相乘后，所有 padding 位置的特征都会变成 0。
        mask_features = token_feature * padding_mask.unsqueeze(-1)
        summed_features = torch.sum(mask_features, dim=1) # 沿seq_len维度求和
        real_lengths = padding_mask.sum(1, keepdim=True) # 每个句子的真实的token长度，去掉填充的
        # torch.clamp 在这里扮演的是一个“安全防护网”的角色，用来防止数学运算中的除以 0 错误。
        # 如果某个句子的全是PAD，那么就会报错，采用torch.clamp()防止数学运算中除以0
        pool_features = summed_features / torch.clamp(real_lengths.float(), min=1e-9)
        logits = self.classifier(pool_features)
        return logits

# 训练与评估
# 将所有与训练、评估、优化和模型保存相关的逻辑都封装到一个Trainer类中, 这个类负责协调模型、数据和优化器，完成整个训练流程。
import os
import json

class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, vaild_loader, device, output_dir="."):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.vaild_loader = vaild_loader
        self.device = device
        self.output_dir = output_dir
        self.best_accuracy = 0.0
        os.makedirs(self.output_dir, exist_ok=True)  # os.makedirs()表示递归创建，os.mkdir()表示只创建最后一级目录，exist_ok=True表示，如果目录存在则跳过，不存在则直接创建
        # 记录历史数据
        self.train_losses = []
        self.valid_losses = []

    def run_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
            self.optimizer.zero_grad()
            token_ids = batch["token_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            output = self.model(token_ids)
            loss = self.criterion(output, labels)
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()

        return total_loss / len(self.train_loader)

    def _evaluate(self, epoch):
        self.model.eval()
        correct_preds = 0
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(self.vaild_loader, desc=f"Epoch {epoch + 1}"):
                # self.optimizer.zero_grad()
                token_ids = batch["token_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(token_ids)
                _, predicted = torch.max(outputs, 1)

                total_samples += labels.size(0)

                correct_preds += (predicted == labels).sum().item()
                # loss = self.criterion(output, labels)
                # total_loss += loss.item()

                # loss.backward()
                # self.optimizer.step()

        return correct_preds / total_samples

    def save_checkpoint(self, epoch, val_accuracy):
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            save_path = os.path.join(self.output_dir, "best_checkpoint.pth")
            torch.save(self.model.state_dict(), save_path)
            print(f"新最佳模型已保存！Epoch: {epoch+1}, 验证集准确率: {val_accuracy}")


    def train(self, epochs, tokenizer, label_map):
        self.train_losses = []
        self.valid_accuracies = []
        for epoch in range(epochs):
            avg_loss = self.run_epoch(epoch)
            val_accuracy = self._evaluate(epoch)

            self.train_losses.append(avg_loss)
            self.valid_accuracies.append(val_accuracy)

            print(f"Epoch {epoch+1} / {epochs} | 训练损失: {avg_loss:.4f} | 验证集: {val_accuracy:.4f}")
            self.save_checkpoint(epoch, val_accuracy)
        print("训练完成！")

        # 训练结束后，保存最终的词典和标签映射
        vocab_path = os.path.join(self.output_dir, "vocab.json")
        with open(vocab_path, "w", encoding='utf-8') as f:
            json.dump(tokenizer.vocab, f, ensure_ascii=False, indent=4)

        label_path = os.path.join(self.output_dir, "label_map.json")
        with open(label_path, "w", encoding='utf-8') as f:
            json.dump(label_map, f, ensure_ascii=False, indent=4)

        print(f"词典{vocab_path}和标签映射{label_path}已保存！")
        return self.train_losses, self.valid_accuracies

# 执行训练过程，定义一个超参数hyperparams管理配置
hyperparams = {
    "vocab_size":len(tokenizer),
    "embed_dim": 128,
    "hidden_dim": 256,
    "num_classes": len(train.target_names),
    "epochs": 20,
    "learning_rate": 0.001,
    "batch_size": 32,
    "device":"cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "output"
}

# 模型实例化
model = TextClassifier(
    hyperparams["vocab_size"],
    hyperparams["embed_dim"],
    hyperparams["hidden_dim"],
    hyperparams["num_classes"],
).to(hyperparams["device"])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
print(hyperparams)


trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=train_loader,
    vaild_loader=valid_loader,
    device=hyperparams["device"],
    output_dir=hyperparams["output_dir"]
)

# 创建标签名->ID映射, 传入trainer 以便保存
label_map = {name: i for i, name in enumerate(train.target_names)}

# 开始训练，并接收返回的历史数据
train_lossed, val_accuracies = trainer.train(epochs=hyperparams["epochs"], tokenizer=tokenizer, label_map=label_map)

# 可视化，绘制每个epoch的损失和验证集准确率
def plot_history(train_losses, val_accuracies, title_prefix=""):
    epochs = range(1, len(train_losses)+1)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 5))

    # 绘制损失曲线
    ax1.plot(epochs, train_losses, 'bo-', label="Train Loss")
    ax1.set_title(f"{title_prefix} Training Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend()

    # 绘制验证集准确率曲线
    ax2.plot(epochs, val_accuracies, 'ro-', label="Validation Accuracy")
    ax2.set_title(f"{title_prefix} Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)
    ax2.legend()

    plt.suptitle(f"{title_prefix} Training Loss and Validation Accuracy", fontsize=20)
    plt.show()

# 绘图函数
plot_history(train_lossed, val_accuracies, title_prefix="Feed-Forward Network")


if __name__ == '__main__':
    pass
