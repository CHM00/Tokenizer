## Llama2架构
Llama2是基于Transformer-Decoder架构的语言模型, 其核心是由N个相同的Transformer-Decoder Block组成的。
Block的主要组件包括:
> **预归一化(Pre-Normalization)**: 与经典Transformer的后归一化不同，Llama2采用预归一化, 输入在进入注意力层和前馈网络之前会先经过一次`RMSNorm`归一化。
> 这有助于提升大模型训练稳定性。

> **自注意力机制(Self-Attention)**: Llama2使用多头自注意力机制来捕捉输入序列中的长程依赖关系。每个注意力头独立地计算注意力分数，然后将它们拼接在一起进行线性变换。
> Llama2支持GQA(Grouped Query Attention)机制, 通过将查询划分为多个组, 每组独立计算注意力（共享键和值）, 提升计算效率。

> **RoPE旋转位置编码(RoPE Positional Encoding)**: Llama2采用RoPE位置编码方法, 通过对查询和键进行旋转变换来引入位置信息, 使模型能够更好地捕捉序列中的相对位置信息。
> 位置并不是在输入端与词向量相加, 而是直接融入到注意力计算中。

> **残差连接(Residual Connection)**: 每个子层（注意力层和前馈网络）都有残差连接, 以缓解深层网络中的梯度消失问题, 并促进信息流动。

整个模型的数据自下而上贯穿所有Transformer Block, 最后通过一个`RMSNorm`归一化层和线性投影层(llm head)输出预测结果。
