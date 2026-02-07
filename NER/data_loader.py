import json
from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def normalize_text(text):
    '''
        规范化文本
    '''
    full_width = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ！＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～＂"
    half_width = r"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&'" + r'()*+,-./:;<=>?@[\]^_`{|}~".'
    mapping = str.maketrans(full_width, half_width)
    return text.translate(mapping)

class Vocabulary:
    '''
        负责管理词汇表 和 token到id 的映射
    '''
    def __init__(self,vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.tokens = json.load(f)
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.pad_id = self.token_to_id.get("<PAD>", 0)
        self.unk_id = self.token_to_id.get("<UNK>", 1)


    def __len__(self):
        return len(self.tokens)

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]




class NerDataset(Dataset):
    def __init__(self, data_path, vocab, tag_map):
        super(NerDataset, self).__init__()
        self.vocab = vocab
        self.tag_to_id = tag_map
        with open(data_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        # 1、根据索引获取原始记录
        record = self.records[idx]
        text =normalize_text(record["text"])
        tokens = list(text)

        # 2、将文本字符转换为 token_ids
        token_ids = self.vocab.convert_tokens_to_ids(tokens)

        # 3、生成与文本等长的tag序列, 默认为'O'
        tags = ['O'] * len(tokens)

        # 4、遍历实体列表, 用 BMES 标签覆盖默认的'O'
        for entity in record.get("entities", []):
            entity_type = entity["type"]
            start = entity["start_idx"]
            end = entity["end_idx"]

            # 跳过无效的实体，因为end是闭区间，能够取到，它的长度是end+1
            if end >= len(tokens):
                continue

            if start == end:
                tags[start] = f"S-{entity_type}"  # 单字实体
            else:
                tags[start] = f"B-{entity_type}"  # 实体开始
                for i in range(start + 1, end):
                    tags[i] = f"M-{entity_type}"  # 实体中间
                tags[end] = f"E-{entity_type}"    # 实体结束

        # 5、将所有实体标签转换为标签ID
        label_ids = [self.tag_to_id[tag] for tag in tags]

        # 6、返回包含两个Tensor的字典
        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "label_ids": torch.tensor(label_ids, dtype=torch.long)
        }

def create_ner_dataloader(data_path, vocab, tag_map, batch_size, shuffle=True):
    dataset = NerDataset(data_path, vocab, tag_map)

    # 在 NLP 任务中，由于每个样本（句子）的长度都不同，所以不能直接让 DataLoader 使用默认的方式打包数据，否则会因序列长度不一而报错。
    # 解决办法: 提供自定义的collate_fn函数, 负责将不同长度的样本进行适当的填充(padding), 以确保每个批次中的所有样本具有相同的长度。
    def collate_fn(batch):
        token_ids_list = [item["token_ids"] for item in batch]
        label_ids_list = [item["label_ids"] for item in batch]
        # pad_sequence 它会找到最长的句子，然后把短的句子用 0 补齐到最长的长度
        # 为什么tag_ids要用-100来填充？因为在计算交叉熵损失时，-100会被忽略，不会影响损失计算

        padded_token_ids = pad_sequence(token_ids_list, batch_first=True, padding_value=vocab.pad_id)
        padded_label_ids = pad_sequence(label_ids_list, batch_first=True, padding_value=-100)
        attention_mask = (padded_token_ids != vocab.pad_id).float()

        return {
            "token_ids": padded_token_ids,
            "label_ids": padded_label_ids,
            "attention_mask": attention_mask
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


if __name__ == "__main__":
    vocab_path = "./data/vocabulary.json"
    train_file = "./data/CMeEE-V2_train.json"
    category_file = "./data/categories.json"

    vocab = Vocabulary(vocab_path)
    print("词汇表的大小:", len(vocab))

    with open(category_file, 'r', encoding='utf-8') as f:
        tag_map = json.load(f)
    train_loader = create_ner_dataloader(train_file, vocab, tag_map, batch_size=4)

    # 3. 验证一个批次的数据
    batch = next(iter(train_loader))

    print("\n--- DataLoader 输出验证 ---")
    print(f"  Token IDs shape: {batch['token_ids'].shape}")
    print(f"  Label IDs shape: {batch['label_ids'].shape}")
    print(f"  Attention Mask shape: {batch['attention_mask'].shape}")
