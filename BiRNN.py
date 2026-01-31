import numpy as np
import torch
import torch.nn as nn

# B T E H 分别表示 批次、序列长度、输入维度以及隐藏维度
B, E, H = 1, 128, 3

def prepare_inputs():
    '''
        使用Numpy准备输入数据
        使用示例句子: "播放周杰伦的<稻香>"
        构造最小词表和随机(可复现)词向量, 生成形状为(B, T, E)
    '''

    np.random.seed(42)
    vocab = {"播放" : 0, "周杰伦": 1, "的": 2, "<稻香>": 3}
    tokens = ["播放", "周杰伦", "的", "<稻香>"]
    ids = [vocab[t] for t in tokens]

    # 词向量表：(V, E)
    V = len(vocab)
    emb_table = np.random.randn(V, E).astype('float32')

    # 取出序列词向量并加上 batch 维度: ( B, T, E )
    x_np = emb_table[ids][None]  # 它在矩阵的最前面增加了一个维度，将形状从 (T, E) 变为 (1, T, E)
    print("x_np.shape: ", x_np.shape)
    return tokens, x_np
prepare_inputs()

def bi_rnn(x_np, U_1, U2, W_1, W_2):
    # 计算x_np
    B, T, _ = x_np.shape

    step_forwards = []
    step_backwards = []

    h_prev = np.zeros((B, H), dtype='float32')
    for t in range(T):
        x_t = x_np[:, t, :]

        h_t = np.tanh(x_t @ U_1 + h_prev @ W_1)
        step_forwards.append(h_t)
        h_prev = h_t

    h_next = np.zeros((B, H), dtype='float32')
    for t in range(T-1, -1, -1):
        x_t = x_np[:, t, :]
        h_t = np.tanh(x_t @ U2 + h_next @ W_2)
        step_backwards.append(h_t)
        h_next = h_t
    step_backwards.reverse()

    # 每个时间步的输出形状变为 (B, 2*H)
    res = np.concatenate([
        np.stack(step_forwards, axis=1),
        np.stack(step_backwards, axis=1)
    ], axis=-1)

    h_n = np.concatenate([step_forwards[-1], step_backwards[0]], axis=1)
    return res, h_n