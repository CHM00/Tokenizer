'''
    模型需要位置信息来理解词元顺序, 传统的位置编码是一种绝对位置编码, 她为每个位置分配一个独立的唯一向量
    Llama2 使用的是相对位置编码, 这种编码方式关注词元之间的相对距离, 而不是它们在序列中的绝对位置
    RoPE (Rotary Position Embedding) 是一种相对位置编码方法, 与传统位置编码不同的是，位置信息不再是加到词嵌入上，而是在计算注意力时，通过复数乘法的方式旋转查询和键的向量表示来引入位置信息。
'''

import torch
# 参数 theta: RoPE 的“基底”，控制位置编码的频率范围，10000.0 是一个标准值。控制每个二维子空间的基础频率w_i, 这个基础频率和位置相乘可以得到旋转的角度。
def precompute_freqs_cis(dim:int, end:int, theta: float=10000.0):
    # 1、计算频率
    freq = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim//2)].float() / dim))

    # 2、计算位置 t = [0, 1, ..., end-1] 生成从 0 到 end-1 的位置索引，对应 RoPE 中的pos
    t = torch.arange(end, device=freq.device)

    # 3、计算频率和位置的外积, 这里计算的就是每个位置对应的旋转角度, [end, dim/2]
    freqs = torch.einsum('i,j->ij', t, freq)

    # 4. 转换为复数形式 (cos(theta) + i*sin(theta))
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 转换成复数, [end ,dim/2]
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # x: [batch_size, seq_len, n_heads, head_dim], 预计算的freqs_cis是[seq_len, head_dim//2]
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)] # [1, seq_len, 1, head_dim//2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 将 Q/K 向量视为复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 准备广播
    freqs_q = reshape_for_broadcast(freqs_cis, xq_)  # 针对 Q 的广播视图

    # 复数乘法即为旋转
    # 在复数域中，两个复数相乘即表示幅角相加、模相乘。由于 freqs_cis 的模为1，这个操作就等价于将 xq_ 向量旋转 freqs_cis 所代表的角度。
    xq_out = torch.view_as_real(xq_ * freqs_q).flatten(3)

    # K 向量可能与 Q 向量有不同的头数（GQA），所以需单独生成广播视图
    freqs_k = reshape_for_broadcast(freqs_cis, xk_)
    xk_out = torch.view_as_real(xk_ * freqs_k).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xq)

# code/C6/llama2/src/rope.py  # 重复相同的KV头, 以匹配查询头的数量
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )




# code/C6/llama2/src/rope.py
if __name__ == "__main__":
    # 准备参数和输入
    batch_size, seq_len, n_heads, n_kv_heads, head_dim = 4, 16, 8, 2, 16
    dim = n_heads * head_dim
    n_rep = n_heads // n_kv_heads

    # --- Test precompute_freqs_cis ---
    print("--- Test precompute_freqs_cis ---")
    freqs_cis = precompute_freqs_cis(dim=head_dim, end=seq_len * 2)
    print("freqs_cis shape:", freqs_cis.shape)

    # --- Test apply_rotary_emb ---
    print("\n--- Test apply_rotary_emb ---")
    xq = torch.randn(batch_size, seq_len, n_heads, head_dim)
    xk = torch.randn(batch_size, seq_len, n_kv_heads, head_dim)
    freqs_cis_slice = freqs_cis[:seq_len]
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis_slice)
    print("xq shape (in/out):", xq.shape, xq_out.shape)
    print("xk shape (in/out):", xk.shape, xk_out.shape)
