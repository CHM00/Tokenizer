'''
    # -*- coding: utf-8 -*-
    NER数据处理：
        模型的输入(X): batch_size x seq_len 其中seq_len为句子长度(词语对应的ID序列)
        模型的输出(y): batch_size x seq_len 其中seq_len代表着对应位置字符的实体标签ID序列
        文本-ID转换：
            （1） 文本->TokenID, 需要构建一个 "字符-ID"的映射表, 即词汇表(Vocabulary)
            （2） 实体->标签ID, 需要构建一个 "标签-ID"的映射表, 即标签表(Label Vocabulary)
'''

import json
import os

def save_to_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 数据结构读取
def collect_entity_types_from_file(file_path):
    types = set()  # 存储实体类型的集合，使用集合避免重复
    with open(file_path, "r", encoding="utf-8") as f:
        datas = json.load(f)

        # 提取实体类型
        for data in datas:
            for entity in data["entities"]:
                types.add(entity["type"])

    return types

# 处理完所有数据文件(训练集、验证集), 以确保包含了全部的实体类型
def generate_tag_map(data_files, output_file):
    all_entity_types = set()
    for file_path in data_files:
        types_in_file = collect_entity_types_from_file(file_path)
        all_entity_types.update(types_in_file)

    # 排序, 保证每次运行结果一致
    sort_types = sorted(list(all_entity_types))

    # 构建 BMES 标签映射, 有了排序后的实体类型列表, 生成对应的标签列表
    '''
        规则如下:
            非实体标签'0'的ID为0
            对于每一种实体类型(dis, dep), 都生成B-dis, M-dis, E-dis, S-dis等4个标签, 并按照顺序分配ID
    '''
    tag_to_id = {"O": 0} # "0"代表非实体标签
    for entity in sort_types:
        for prefix in ["B", "M", "E", "S"]:
            tag_name = f"{prefix}-{entity}"
            tag_to_id[tag_name] = len(tag_to_id)

    print(f"\n已生成 {len(tag_to_id)} 个标签映射")

    # 保存标签映射表
    # save_path = "./data/tag_to_id.json"
    save_to_json(tag_to_id, output_file)
    print(f"标签映射表已保存到: {output_file}\n")

    return tag_to_id


# 构建词汇表， 创建一个“字符-ID”的映射表（词汇表）
# 获取数据中出现的所有字符
from collections import Counter
import json

def create_char_vocab(data_files, out_file, min_freq=1):
    char_counts = Counter()
    with open(data_files, "r", encoding="utf-8") as f:
        all_data = json.load(f)
        for data in all_data:
            normalized_text = normalize_text(data["text"])
            char_counts.update(normalized_text)

    # 过滤低频词
    frequent_chars = [char for char, count in char_counts.items() if count >= min_freq]

    # 排序, 确保每次生成词汇表的文件内容一致
    frequent_chars.sort()

    # 添加特殊Token
    special_Tokens = ["<PAD>", "<UNK>"]
    final_vocab_list = special_Tokens + frequent_chars

    print("词汇表大小:", len(final_vocab_list))

    save_to_json(final_vocab_list, out_file)
    print("保存词汇表到:", out_file)


    print(f"初步统计的字符种类数: {len(char_counts)}")


# 规范化文本: 数据中可能同时包含全角字符(如：，。！？)和半角字符(如: ,.!?)，他们语义上相同, 但是会被视为两个不同的Token, 将所有全角字符转换为半角字符
def normalize_text(text):
    '''
        规范化文本
    '''
    full_width = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ！＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～＂"
    half_width = r"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&'" + r'()*+,-./:;<=>?@[\]^_`{|}~".'
    mapping = str.maketrans(full_width, half_width)
    return text.translate(mapping)




if __name__ == "__main__":
    train_file_path = "./data/CMeEE-V2_train.json"
    # entity_types = collect_entity_types_from_file(file_path)
    # print("实体类型集合:", entity_types)
    # print("实体类型数量:", len(entity_types))
    dev_file_path = "./data/CMeEE-V2_dev.json"
    output_file = "./data/categories.json"
    data_files = [train_file_path, dev_file_path]
    tag_to_id = generate_tag_map(data_files, output_file)
    # print("实体类型集合:", entity_types)
    # print("实体类型数量:", len(entity_types))
    print("标签映射表:", tag_to_id)
    output_path = './data/vocabulary.json'
    create_char_vocab(train_file_path, output_path)