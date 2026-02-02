import torch
import json
import os
from lstm_classfication import hyperparams, Tokenizer, TextClassifierBasedLSTM

# 推理过程
class Predictor:
    def __init__(self, model, tokenizer, label_map, device, max_len=128):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.id_to_label = {idx: label for label, idx in self.label_map.items()}
        self.device = device
        self.max_len = max_len

    def predict(self, text):
        token_ids = self.tokenizer.tokenize(text)
        chunks = []
        if len(token_ids) > self.max_len:
            stride = max(1, int(self.max_len * 0.8))
            for i in range(0, len(token_ids) - self.max_len + 1, stride):
                chunks.append(token_ids[i:i + self.max_len])
        else:
            chunks.append(token_ids)

        # 手动计算长度并进行填充
        chunk_lengths = [len(c) for c in chunks]
        max_chunk_len = max(chunk_lengths) if chunk_lengths else 0

        padded_chunks = []
        for chunk in chunks:
            padding_len = max_chunk_len - len(chunk)
            padded_chunks.append(chunk + [0] * padding_len)

        if not padded_chunks:
            return "无法预测（文本过短）"

        chunk_tensors = torch.tensor(padded_chunks, dtype=torch.long).to(self.device)
        length_tensors = torch.tensor(chunk_lengths, dtype=torch.long)  # 长度在CPU上

        # chunk_tensors = torch.tensor(chunks, dtype=torch.long).to(self.device)
        with torch.no_grad():
            outputs = self.model(chunk_tensors, length_tensors)
            preds = torch.argmax(outputs, dim=1)  # 直接返回的最大值的类别索引

        # torch.bincount()统计每个类别ID出现的次数, .argmax()找到次数最多的那个位置, 将ID转换为人类可读的标签
        final_pred_id = torch.bincount(preds).argmax().item()  # 对于无法一次读取超长内容, 将其切分成多个小块，查看多个小块的预测结果，找出出现次数最多的类别，作为整篇文档的最终答案
        final_pred_label = self.id_to_label[final_pred_id]
        return final_pred_label

# 加载资源
vocab_path = os.path.join(hyperparams["output_dir"], "vocab.json")
with open(vocab_path, "r", encoding='utf-8') as f:
    loaded_vocab = json.load(f)

label_path = os.path.join(hyperparams["output_dir"], "label_map.json")
with open(label_path, "r", encoding='utf-8') as f:
    loaded_label_map = json.load(f)

# 实例化推理组件
inference_tokenizer = Tokenizer(vocab=loaded_vocab)
inference_model = TextClassifierBasedLSTM(
    len(inference_tokenizer),
    hyperparams["embed_dim"],
    hyperparams["hidden_dim"],
    len(loaded_label_map),
).to(hyperparams["device"])

model_path = os.path.join(hyperparams["output_dir"], "best_checkpoint.pth")
inference_model.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))

predictor = Predictor(inference_model, inference_tokenizer, loaded_label_map, device=hyperparams["device"])

new_text = "The doctor prescribed a new medicine for the patient's illness, focusing on its gpu accelerated healing properties."
predicted_class = predictor.predict(new_text)

print("new text:", new_text)
print("predicted class:", predicted_class)
