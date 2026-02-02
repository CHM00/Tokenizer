from sklearn.datasets import fetch_20newsgroups
from torch import no_grad

# 选取四个类别的数据
col = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
train = fetch_20newsgroups(subset='train',categories=col,shuffle=True, random_state=42)
test = fetch_20newsgroups(subset='test',categories=col,shuffle=True, random_state=42)

import re
# 分词函数
def basic_tokenize(text):
    '''
        基于正则表达式的基本分词函数, 分割单词
    '''
    text.lower()
    text = re.sub(r"[^a-z0-9(),.!?\\'`]", " ", text)
    text = re.sub(r"([,.!?\\'`])", r" \\1 ", text)
    text = text.strip().split()
    return text

from collections import Counter
# 获取token
word_count = Counter()
for text in train.data:
    tokens = basic_tokenize(text)
    word_count.update(tokens)

print("词汇表大小: ", len(word_count))

# 构建词汇表
def build_vocab(word_count, min_freq=5):
    vocab = {"<PAD>":0, "<UNK>":1}
    for word, count in word_count.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(word_count, min_freq=5)
print("词表大小: ", len(vocab))


# 定义一个Tokenizer
class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token_to_id = {token: id for token, id in vocab.items()}

    def basic_tokenize(self, text):
        '''
            基于正则表达式的基本分词函数, 分割单词
        '''
        text.lower()
        text = re.sub(r"[^a-z0-9(),.!?\\'`]", " ", text)
        text = re.sub(r"([,.!?\\'`])", r" \\1 ", text)
        text = text.strip().split()
        return text

    def tokenize(self, text):
        '''
            获取token_ids, 网络中只接受数字
        '''
        tokens = self.basic_tokenize(text)
        token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        return token_ids

    def __len__(self):
        return len(self.vocab)

tokenizer = Tokenizer(vocab)
print("Vocabulary Size: ", len(tokenizer))

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
            token_ids = self.tokenizer.tokenize(text)

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


import random

# 构建一个新的Dataset类，增加训练时的随机遮盖功能，随机遮盖一些token_ids, 以增强模型的鲁棒性
class TextClassificationDatasetWithMasking(TextDataset):
    def __init__(self, texts, labels, tokenizer, max_len=128, is_train=False, mask_prob=0.1):
        super().__init__(texts, labels, tokenizer, max_len)
        self.is_train = is_train
        self.mask_prob = mask_prob
        self.unk_token_id = tokenizer.token_to_id.get("<UNK>", 1)

    def __getitem__(self, idx):
        # 关键：创建副本，避免修改原始数据
        item = super().__getitem__(idx).copy()

        if self.is_train:
            token_ids = item['token_ids']
            masked_token_ids = []
            for token_id in token_ids:
                # 不遮盖PAD (ID=0)
                if token_id != 0 and random.random() < self.mask_prob:
                    masked_token_ids.append(self.unk_token_id)
                else:
                    masked_token_ids.append(token_id)
            item['token_ids'] = masked_token_ids

        return item



def collate_fn(batch):
    '''
        负责处理一个批次内长短不一的样本，通过填充统一长度
    '''
    max_batch_len = max(len(item["token_ids"]) for item in batch)

    batch_token_ids, batch_labels, batch_lengths = [], [], []

    for item in batch:
        token_ids = item["token_ids"]
        length = len(token_ids)
        padding_len = max_batch_len - length

        padded_ids = token_ids + ([0] * padding_len)
        batch_token_ids.append(padded_ids)
        batch_labels.append(item["label"])
        batch_lengths.append(length)
    return {
        "token_ids": torch.tensor(batch_token_ids, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "lengths": torch.tensor(batch_lengths, dtype=torch.long)
    }

# 定义文本分类模型
class TextClassifierBasedLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, hidden_dim, n_layers=1, dropout=0.3, bidirectional=False):
        super(TextClassifierBasedLSTM, self).__init__()
        self.hidden_size = hidden_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=dropout,
            num_layers=n_layers,
            bidirectional=bidirectional
        )

        num_direction = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_direction, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, token_ids, lengths):
        embedding = self.embedding(token_ids)

        # 打包序列
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedding,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # lstm前向传播
        # hidden和cell分别是最后一个时间步的隐藏状态和细胞状态，形状为 (num_layers * num_directions, batch_size, hidden_size)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # 提取最终隐藏状态用于分类
        if self.lstm.bidirectional:
            # 拼接最后一个时间步的前向和后向的隐藏状态
            # hidden[-2, :, :] 是前向的最后一个隐藏状态
            # hidden[-1, :, :] 是后向的最后一个隐藏状态
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            # 只取最后一层的隐藏状态
            hidden = hidden[-1,:,:]
        output = self.classifier(hidden)
        return output

# 训练与评估
# 将所有与训练、评估、优化和模型保存相关的逻辑都封装到一个Trainer类中, 这个类负责协调模型、数据和优化器，完成整个训练流程。
import os
import json

class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, valid_loader, device, output_dir="."):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.output_dir = output_dir
        self.best_accuracy = 0.0
        os.makedirs(self.output_dir, exist_ok=True)  # os.makedirs()表示递归创建，os.mkdir()表示只创建最后一级目录，exist_ok=True表示，如果目录存在则跳过，不存在则直接创建
        # 记录历史数据
        self.train_losses = []
        self.valid_accuracies = []

    def run_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
            self.optimizer.zero_grad()
            token_ids = batch["token_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            lengths = batch["lengths"]
            output = self.model(token_ids, lengths)
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
            for batch in tqdm(self.valid_loader, desc=f"Epoch {epoch + 1}"):
                # self.optimizer.zero_grad()
                token_ids = batch["token_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                lengths = batch["lengths"]
                outputs = self.model(token_ids,lengths)
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


import os
import json


class TrainerWithEarlyStopping(Trainer):
    def __init__(self, model, optimizer, criterion, train_loader, valid_loader, device, output_dir=".", patience=3):
        super().__init__(model, optimizer, criterion, train_loader, valid_loader, device, output_dir)
        self.patience = patience
        self.epochs_no_improve = 0

    def train(self, epochs, tokenizer, label_map):
        for epoch in range(epochs):
            avg_loss = self.run_epoch(epoch)
            val_accuracy = self._evaluate(epoch)

            self.train_losses.append(avg_loss)
            self.valid_accuracies.append(val_accuracy)

            print(f"Epoch {epoch + 1}/{epochs} | 训练损失: {avg_loss:.4f} | 验证集准确率: {val_accuracy:.4f}")

            current_best = self.best_accuracy
            self.save_checkpoint(epoch, val_accuracy)

            if self.best_accuracy > current_best:
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print(f"\n提前停止于 Epoch {epoch + 1}，因为验证集准确率连续 {self.patience} 轮未提升。")
                break

        print("\n训练完成！")
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
    "output_dir": "output_lstm",
    "n_layers": 2,          # 新增
    "dropout": 0,           # 新增：此处显式设为 0，当前不启用 Dropout
    "bidirectional": True,  # 新增
}
import matplotlib.pyplot as plt

# # 模型实例化
# model = TextClassifierBasedLSTM(
#     hyperparams["vocab_size"],
#     hyperparams["embed_dim"],
#     hyperparams["num_classes"],
#     hyperparams["hidden_dim"],
#     n_layers=hyperparams["n_layers"],
#     dropout=hyperparams["dropout"],
#     bidirectional=hyperparams["bidirectional"]
# ).to(hyperparams["device"])
#
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
# print(hyperparams)

from torch.utils.data import DataLoader
#
# train_dataset = TextDataset(train.data, train.target, tokenizer, max_len=128)
# test_dataset = TextDataset(test.data, test.target, tokenizer, max_len=128)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
# valid_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# 构建带有数据增强的Dataset和DataLoader
# 训练集使用新的Dataset，并启动is_train和mask_prob参数
train_dataset_with_masking = TextClassificationDatasetWithMasking(
    train.data,
    train.target,
    tokenizer,
    max_len=128,
    is_train=True,
    mask_prob=0.1
)
test_dataset_with_masking = TextClassificationDatasetWithMasking(
    test.data,
    test.target,
    tokenizer,
    max_len=128,
    is_train=False,
    mask_prob=0.1
)

train_loader = DataLoader(train_dataset_with_masking, batch_size=32, shuffle=True, collate_fn=collate_fn)
# 验证集使用创建好的不带Masking的dataset
valid_loader = DataLoader(test_dataset_with_masking, batch_size=32, shuffle=True, collate_fn=collate_fn)
hyperparams_reg = hyperparams.copy()
hyperparams_reg["output_dir"] = "output_lstm_with_early_stopping"



# 模型实例化
model_reg = TextClassifierBasedLSTM(
    hyperparams_reg["vocab_size"],
    hyperparams_reg["embed_dim"],
    hyperparams_reg["num_classes"],
    hyperparams_reg["hidden_dim"],
    n_layers=hyperparams_reg["n_layers"],
    dropout=0.3,
    bidirectional=hyperparams_reg["bidirectional"]
).to(hyperparams_reg["device"])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_reg.parameters(), lr=hyperparams_reg["learning_rate"])
print(hyperparams_reg)

trainer = TrainerWithEarlyStopping(
    model=model_reg,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=train_loader,
    valid_loader=valid_loader,
    device=hyperparams_reg["device"],
    output_dir=hyperparams_reg["output_dir"],
    patience=3
)

# trainer = Trainer(
#     model=model,
#     optimizer=optimizer,
#     criterion=criterion,
#     train_loader=train_loader,
#     vaild_loader=valid_loader,
#     device=hyperparams["device"],
#     output_dir=hyperparams["output_dir"]
# )

# 创建标签名->ID映射, 传入trainer 以便保存
label_map = {name: i for i, name in enumerate(train.target_names)}

# 开始训练，并接收返回的历史数据
train_lossed, val_accuracies = trainer.train(epochs=hyperparams_reg["epochs"], tokenizer=tokenizer, label_map=label_map)

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