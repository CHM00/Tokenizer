'''
    门的结构：让信息选择性通过的结构, 类似于神经网络中的开关
        输入: 当前时间步的输入x_t和上一时间步的隐藏状态h_{t-1}的拼接 -- RNN的原理也是这样, 基于前一时刻的隐藏状态和当前输入来计算当前隐藏状态
        计算: 对输入进行线性变换，然后通过激活函数（通常是sigmoid）来生成一个在0到1之间的值
        输出: 一个元素值在(0,1)之间的向量, 这个向量将与另外一个向量进行按元素的乘法
            当门输出向量的某个元素接近1时，表示允许对应维度的信息完全通过
            当门输出向量的某个元素接近0时，表示阻止对应维度的信息通过
        输入门保护细胞状态不受无关输入的干扰
        输出门保护其他单元不受当前细胞状态中无关记忆的干扰

    LSTM的组成:
        遗忘门：遗忘cell状态中的哪些信息
        输入门：决定要将哪些新信息添加到cell中
        候选记忆： 从当前输入和上一个隐藏状态中创建新的候选记忆
        细胞状态更新：结合遗忘门和输入门的输出（候选记忆和输入门决定要添加的新信息），更新cell状态
        输出门：决定从cell中输出哪些信息
        隐藏状态更新：结合输出门和更新后的cell状态，生成新的隐藏状态
'''
import numpy as np

B, T, H = 1, 4, 3
E = 128

def prepare_inputs():
    vocab = {"播放" : 0, "周杰伦": 1, "的": 2, "<稻香>": 3}
    tokens = ["播放", "周杰伦", "的", "<稻香>"]

    # 获取token_id
    ids = [vocab[t] for t in tokens]

    # 词向量表示
    np.random.seed(42)
    V = len(vocab)

    emb_table = np.random.randn(V, E).astype('float32')
    x_np = emb_table[ids][None]  # (B, T, E)
    return tokens, x_np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def manual_lstm_numpy(x_np, weights):
    # 其中U_c、W_c是用于计算候选记忆的权重矩阵
    (U_f, W_f, U_i, W_i, U_c, W_c, U_o, W_o) = weights

    B, T, _ = x_np.shape
    steps = []

    # 历史隐藏状态
    h_prev = np.zeros((B, H), dtype='float32')
    c_prev = np.zeros((B, H), dtype='float32')
    for t in range(T):
        x_t = x_np[:, t, :]
        # 遗忘门 @是矩阵乘法
        O_f = sigmoid(x_t @ U_f + h_prev @ W_f)
        # 输入门
        O_i = sigmoid(x_t @ U_i + h_prev @ W_i)
        # 候选记忆
        C_tilde = np.tanh(x_t @ U_c + h_prev @ W_c)
        # 输出门
        O_o = sigmoid(x_t @ U_o + h_prev @ W_o)

        # 细胞状态更新
        c_t = O_f * c_prev + O_i * C_tilde  # 按元素乘法和按元素加法

        # 隐藏状态更新
        h_t = O_o * np.tanh(c_t)
        steps.append(h_t)
        h_prev = h_t
        c_prev = c_t
    res = np.stack(steps, axis=1)
    return res, h_prev, c_prev


def main():
    tokens, x_np = prepare_inputs()

    np.random.seed(42)
    U_f = np.random.randn(E, H).astype('float32')
    U_i = np.random.randn(E, H).astype('float32')
    U_c = np.random.randn(E, H).astype('float32')
    U_o = np.random.randn(E, H).astype('float32')
    W_f = np.random.randn(H, H).astype('float32')
    W_i = np.random.randn(H, H).astype('float32')
    W_o = np.random.randn(H, H).astype('float32')
    W_c = np.random.randn(H, H).astype('float32')
    weights = (U_f, W_f, U_i, W_i, U_c, W_c, U_o, W_o)
    # res, h_prev, c_prev = manual_lstm_numpy(x_np, weights)
    print("------手写LSTM（NUMPY）----------")
    res, h_prev, c_prev = manual_lstm_numpy(x_np, weights)
    print("x_np.shape: ", x_np.shape)
    print("res.shape: ", res.shape)
    print("h_prev.shape: ", h_prev.shape)
    print("c_prev.shape: ", c_prev.shape)

if __name__ == "__main__":
    print("------LSTM实现----------")
    main()