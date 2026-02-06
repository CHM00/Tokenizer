'''
    SwiGLU FeedForward Module Implementation
'''
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))



class MoE(nn.Module):
    def __init__(self, dim:int, hidden_dim:int, multiple_of:int, ffn_dim_multiplier: Optional[float], num_experts: int=8, top_k: int=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # 门控网络, 决定每个token去往哪个专家
        self.gate = nn.Linear(dim, self.num_experts, bias=False)
        # 专家网络列表, 创建num_experts个FFN模块
        self.experts = nn.ModuleList([
            FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq, dim)
        # batch, seq, dim = x.shape
        B, T, D = x.shape
        x_flat = x.view(-1, D)

        # 门控网络计算专家分配概率
        gate_logits = self.gate(x_flat)  # (B*T, num_experts)

        # TopK路由
        weights, indices = torch.topk(gate_logits, k=self.top_k, dim=-1)
        # 归一化权重
        weights = F.softmax(weights, dim=-1)

        # 创建一个形状、数据类型（dtype）、设备（CPU/GPU）都与 x_flat 完全一样的全零张量。
        # 在 MoE 中，Token 被分发给不同的专家计算后，结果是散乱的。我们需要一个容器，把计算完的结果按原来的索引位置填回去。
        output = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            '''
                逻辑是：每个token找专家，实现过程是每个专家认领token -> 批量计算 -> 加权累加回output
                表示哪些token选择了这个专家，找到对应的token, 然后获取其输出, 考虑专家对每个token的贡献不一样(权重不同)，最后把加权后的结果累加回output中
            '''
            # 找到所有选中当前专家i的token索引
            batch_idx, k_idx = torch.where(indices == i)  # batch_idx: (n_selected,) k_idx: (n_selected,)

            if len(batch_idx) == 0:
                continue  # 如果没有token被分配给这个专家，跳过

            # 取出对应的输入进行计算
            expert_input = x_flat[batch_idx]  # (n_selected, dim)
            expert_out = expert(expert_input) # (n_selected, dim)

            # 获取对应的权重
            expert_weights = weights[batch_idx, k_idx].unsqueeze(-1) # (num_selected, 1)

            # 6. 将结果加权累加回输出张量, 0表示操作第0维, batch_idx表示行索引, 往哪一行上加, expert_out * expert_weights表示加的值, 防止覆盖
            # 因为 MoE 通常是 Top-K (K>1)。这意味着一个 Token 会被送给多个专家
            output.index_add_(0, batch_idx, expert_out * expert_weights)
        return output.view(B, T, D)


# if __name__ == "__main__":
    # # 准备参数和输入
    # batch_size, seq_len, dim = 4, 16, 128
    #
    # # 初始化 FFN 模块
    # ffn = FeedForward(
    #     dim=dim,
    #     hidden_dim=4 * dim,
    #     multiple_of=256,
    #     ffn_dim_multiplier=None
    # )
    #
    # # 准备输入
    # x = torch.randn(batch_size, seq_len, dim)
    #
    # # 执行前向传播
    # output = ffn(x)
    #
    # # 验证输出形状
    # print("--- FeedForward (SwiGLU) Test ---")
    # print("Input shape:", x.shape)
    # print("Output shape:", output.shape)

if __name__ == "__main__":
    # --- 超参数设置 ---
    BATCH_SIZE = 2  # 比如你有 2 个化工过程样本
    SEQ_LEN = 5  # 时间步长 (Lookback window)
    DIM = 16  # 特征维度
    HIDDEN_DIM = 32  # FFN 隐藏层维度
    NUM_EXPERTS = 4  # 专家数量
    TOP_K = 2  # 每个 Token 选 2 个专家

    print(f"--- 配置 ---")
    print(f"Input Shape: ({BATCH_SIZE}, {SEQ_LEN}, {DIM})")
    print(f"Experts: {NUM_EXPERTS}, TopK: {TOP_K}")
    print("-" * 30)

    # 1. 实例化模型
    moe_model = MoE(
        dim=DIM,
        hidden_dim=HIDDEN_DIM,
        multiple_of=4,
        ffn_dim_multiplier=None,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K
    )

    # 2. 创建模拟输入数据 (随机数)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, DIM)
    print("输入数据 x 的形状:", x.shape)

    # 3. 前向传播
    y = moe_model(x)

    # 4. 验证输出
    print("输出数据 y 的形状:", y.shape)

    # 5. 简单验证是否运行成功
    if x.shape == y.shape:
        print("\n 成功运行！输出形状与输入形状一致。")
    else:
        print("\n 形状不匹配，请检查代码。")

    # --- 查看一下路由情况 ---
    print("\n--- 调试视角：看看每个专家处理了多少 Token ---")
    x_flat = x.view(-1, DIM)
    gate_logits = moe_model.gate(x_flat)
    _, indices = torch.topk(gate_logits, k=TOP_K, dim=-1)

    total_tokens = BATCH_SIZE * SEQ_LEN * TOP_K  # 因为每个Token去2个地方，所以总处理次数翻倍
    print(f"总计算请求次数 (Tokens * TopK): {total_tokens}")

    for i in range(NUM_EXPERTS):
        # 统计索引中等于当前专家ID i 的次数
        count = (indices == i).sum().item()
        print(f"专家 {i} 被调用的次数: {count}")
