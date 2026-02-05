import torch
import torch.nn as nn

from collections import Counter
from sklearn.datasets import fetch_20newsgroups


# 选取四个类别的数据
col = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
train = fetch_20newsgroups(subset='train',categories=col,shuffle=True, random_state=42)
test = fetch_20newsgroups(subset='test',categories=col,shuffle=True, random_state=42)

# import re
#
# def tokenizer(text):
#     text.lower()
#     text = re.sub(r"[^a-z0-9(),.!?\\'`]", " ", text)
#     text = re.sub(r"[,.!?\\'`]", r" \\1 ", text)
#     text = text.strip().split()
#     return text
#
# word_count = Counter()
# for text in train.data:
#     token = tokenizer(text)
#     word_count.update(token)
#
# print("词汇表大小: ", len(word_count))
#
#
#
# # 构建一个词汇表
# def build_vocab(word_count, min_freq=5):
#     vocab={"<PAD>": 0, "<UNK>": 1}
#     for word, count in word_count.items():
#         if count >= min_freq:
#             vocab[word] = len(vocab)
#     return vocab
# vocab = build_vocab(word_count, min_freq=5)
#
# print("词表大小: ", len(vocab))
#
# # Tokenizer 类用于将文本转换为词汇表索引序列
# class Tokenizer:
#     def __init__(self, vocab):
#         self.vocab = vocab
#         self.token_to_id = {token: idx for token, idx in vocab.items()}
#
#     def text_to_token(self, text):
#         text.lower()
#         text = re.sub(r"[^a-z0-9(),.!?\\'`]", " ", text)
#         text = re.sub(r"[,.!?\\'`]", r" \\1 ", text)
#         text = text.strip().split()
#         return text
#
#     def _tokenizer(self, text):
#         '''
#             模型只接受数字索引输入，需要将文本转换为对应的索引序列
#         '''
#         tokens = self.text_to_token(text)
#         token_ids = [self.token_to_id.get(token, self.vocab["<UNK>"]) for token in tokens]
#         return token_ids
#
#     def __len__(self):
#         return len(vocab)
#
# tokenizer = Tokenizer(vocab)
# print("Vocabulary size:", len(tokenizer))
#
#
# from torch.utils.data import Dataset
# from tqdm import tqdm
# from torch.utils.data import DataLoader
#
# class TextDataset(Dataset):
#     '''
#         接受原始文本，调用tokenzier进行ID化，并生成适合模型输入的张量数据，应用滑动分割策略处理长文本。如果文本超过max_len, 则会进行切分。
#     '''
#     def __init__(self, texts, labels, tokenizer, max_len=128):
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.processed_data=[]
#
#         for text, label in tqdm(zip(texts, labels), total=len(labels)):
#             # 文本转换为token, 进行ID化
#             token_ids = self.tokenizer.tokenize(text)
#
#             if len(token_ids) > self.max_len:
#                 stride = max(1, int(self.max_len * 0.8))
#                 for i in range(0, len(token_ids)- self.max_len + 1, stride):
#                     chunk = token_ids[i:i+self.max_len]
#                     self.processed_data.append({"token_ids": chunk, "label": label})
#             else:
#                 self.processed_data.append({"token_ids": token_ids, "label": label})
#
#     def __len__(self):
#         return len(self.processed_data)
#
#     def __getitem__(self, idx):
#         return self.processed_data[idx]
#
#
# def collate_fn(batch):
#     '''
#         负责处理一个批次内长短不一的样本，通过填充统一长度
#     '''
#     max_batch_len = max(len(item["token_ids"]) for item in batch)
#
#     batch_token_ids, batch_labels, batch_lengths = [], [], []
#
#     for item in batch:
#         token_ids = item["token_ids"]
#         length = len(token_ids)
#         padding_len = max_batch_len - length
#
#         padded_ids = token_ids + ([0] * padding_len)
#         batch_token_ids.append(padded_ids)
#         batch_labels.append(item["label"])
#         # batch_lengths.append(length)
#     return {
#         "token_ids": torch.tensor(batch_token_ids, dtype=torch.long),
#         "labels": torch.tensor(batch_labels, dtype=torch.long)
#     }
#
# train_dataset = TextDataset(train.data, train.target, tokenizer, max_len=128)
# test_dataset = TextDataset(test.data, test.target, tokenizer, max_len=128)
#
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
# valid_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)



import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# 查看特殊token
print(f"UNK token: '{tokenizer.unk_token}', ID: {tokenizer.unk_token_id}")
print(f"PAD token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
print(f"CLS token: '{tokenizer.cls_token}', ID: {tokenizer.cls_token_id}")
print(f"SEP token: '{tokenizer.sep_token}', ID: {tokenizer.sep_token_id}")
print(f"Vocab size: {tokenizer.vocab_size}")

# Bert的tokenizer会自动处理文本的预处理(如小写转换、标点分割等)，并为文本添加特殊的CLS和SEP标记
# [CLS]位于序列开头, 它在BERT输出中对应的向量通常被用作整个序列的聚合表示，非常适合用于分类任务
# [SEP]用于分隔两个句子, 在单句分类任务中则表示句子的结束

from tqdm import tqdm


class BertTextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer: BertTokenizer, max_len=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.process_data = []

        for text, label in tqdm(zip(self.texts, self.labels), total= len(self.labels), desc="Processing Dataset"):
            tokens = self.tokenizer(text)
            if len(tokens["input_ids"]) > self.max_len:
                stride = max(1, int(self.max_len * 0.8))
                for i in range(0, len(tokens["input_ids"]) - self.max_len + 1, stride):
                    chunk = tokens["input_ids"][i:i + self.max_len]
                    self.process_data.append(
                        {"token_ids": chunk, "label": label}
                    )
            else:
                self.process_data.append(
                    {"token_ids": tokens["input_ids"], "label": label}
                )

    def __len__(self):
        return len(self.process_data)

    def __getitem__(self, idx):
        return self.process_data[idx]

'''
    BERT的一个重要输入是attention_mask(注意力掩码)，它指示模型哪些token是实际内容，哪些是填充部分。
    模型根据这个掩码在计算注意力时忽略填充部分，从而避免填充对模型输出产生不必要的影响。
'''
def bert_collate_fn(batch):
    max_batch_len = max(len(item["token_ids"]) for item in batch)

    batch_input_ids, batch_attention_mask, batch_labels = [], [], []

    for item in batch:
        input_ids = item["token_ids"]
        # 需要填充的padding长度
        padding_length = max_batch_len - len(input_ids)

        padded_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        # 生成attention_mask，实际内容为1，填充部分为0
        attention_mask = [1] * len(input_ids) + [0] * padding_length

        batch_input_ids.append(padded_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(item["label"])

    return {
        "token_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long)
    }

# 构建文本分类的bert模型
from transformers import BertModel
import torch.nn as nn
class TextClassificationBERT(nn.Module):
    def __init__(self, model_name, num_classes, freeze_bert=False):
        super().__init__()
        # 1、加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(model_name)
        # 2、定义分类头, 分类头的输入维度从self.bert.config.hidden_size中获取
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        # 3、是否冻结BERT参数
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # 1、获取BERT的输出
        outputs= self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 2、提取[CLS]标记的向量表示
        cls_output = outputs.pooler_output  # shape: (batch_size, hidden_size)
        # 3、通过分类头得到最终的类别预测
        logits = self.classifier(cls_output)  # shape: (batch_size, num_classes)
        return logits

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
            attention_mask = batch["attention_mask"].to(self.device)
            output = self.model(token_ids, attention_mask)
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
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(token_ids, attention_mask)
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
            # 对于transformers模型，推荐使用save_pretrained来保存
            self.model.bert.save_pretrained(self.output_dir)
            # 训练脚本中
            tokenizer.save_pretrained(hparams["output_dir"])
            # 单独保存分类头
            classifier_path = os.path.join(self.output_dir, "classifier.pth")
            torch.save(self.model.classifier.state_dict(), classifier_path)
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


hparams = {
    "model_name": 'bert-base-uncased',
    "num_classes": len(train.target_names),
    "freeze_bert": False,
    "epochs": 5,             # 减少轮次
    "learning_rate": 2e-5,   # 降低学习率
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "output_bert"
}
if __name__ == "__main__":

    train_dataset = BertTextClassificationDataset(train.data, train.target, tokenizer, max_len=128)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=bert_collate_fn)

    test_dataset = BertTextClassificationDataset(test.data, test.target, tokenizer, max_len=128)
    valid_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=bert_collate_fn)



    model = TextClassificationBERT(
        hparams["model_name"],
        hparams["num_classes"],
    ).to(hparams["device"])


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])
    print(hparams)

    import matplotlib.pyplot as plt

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        train_loader,
        valid_loader,
        hparams["device"],
        hparams["output_dir"]
    )

    # 创建标签名->ID映射, 传入trainer 以便保存
    label_map = {name: i for i, name in enumerate(train.target_names)}

    # 开始训练，并接收返回的历史数据
    train_lossed, val_accuracies = trainer.train(epochs=hparams["epochs"], tokenizer=tokenizer, label_map=label_map)


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


