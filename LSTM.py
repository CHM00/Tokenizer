'''
    LSTM的组成:
        遗忘门：遗忘cell状态中的哪些信息
        输入门：决定要将哪些新信息添加到cell中
        候选记忆： 从当前输入和上一个隐藏状态中创建新的候选记忆
        细胞状态更新：结合遗忘门和输入门的输出（候选记忆和输入门决定要添加的新信息），更新cell状态
        输出门：决定从cell中输出哪些信息
        隐藏状态更新：结合输出门和更新后的cell状态，生成新的隐藏状态
'''
import numpy as np

def manual_lstm_numpy(x_np, weights):
    U_f, W_f, U_i, W_i, U_c, W_c, U_o, W_o = weights