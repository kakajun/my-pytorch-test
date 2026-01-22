"""
========================================================================================
项目名称: Transformer 核心架构实现 (The Transformer)
论文来源: "Attention Is All You Need" (Google, 2017)
任务演示: 序列复制任务 (Sequence Copy Task)
========================================================================================

【架构详解】
Transformer 是现代 NLP (如 BERT, GPT) 的基石，它完全抛弃了 RNN/LSTM 的循环结构，
完全依赖 **Self-Attention (自注意力机制)** 来捕捉序列中的长距离依赖。

主要由以下组件构成：

1. **编码器 (Encoder)**
   - 由 N 个 EncoderLayer 堆叠而成。
   - 每个层包含两个子层：
     a. **Multi-Head Self-Attention**: 让模型关注句子中不同位置的信息（捕捉上下文）。
     b. **Position-wise Feed-Forward**: 全连接前馈网络，用于特征变换。
   - 包含 **Residual Connection (残差连接)** 和 **Layer Normalization (层归一化)**。

2. **解码器 (Decoder)**
   - 由 N 个 DecoderLayer 堆叠而成。
   - 每个层包含三个子层：
     a. **Masked Multi-Head Self-Attention**:
        带掩码的自注意力，确保预测第 i 个词时只能看到 i 之前的词（防止偷看未来）。
     b. **Multi-Head Cross-Attention**:
        交叉注意力，Query 来自解码器，Key/Value 来自编码器（用于对齐源语言和目标语言）。
     c. **Position-wise Feed-Forward**.

3. **关键技术点**
   - **Positional Encoding (位置编码)**:
     因为 Transformer 没有循环结构，无法识别顺序，必须手动注入位置信息（正弦/余弦函数）。
   - **Scaled Dot-Product Attention**:
     注意力核心公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
   - **Parallelism**:
     相比 RNN，Transformer 可以并行计算整个序列，训练速度极大提升。

【当前代码演示】
实现了一个完整的 Transformer 模型，并训练它完成一个简单的“数字序列复制”任务。
输入: [1, 5, 2, 8] -> 输出: [1, 5, 2, 8]
这证明了模型具备捕捉序列模式并生成对应输出的能力。
========================================================================================
"""
import torch
import torch.nn as nn
import torch.optim as optim
import math
import copy

# ==========================================
# 1. 基本组件：多头注意力机制 (Multi-Head Attention)
# ==========================================


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # 计算注意力分数: (batch, heads, seq_len, seq_len)
        attn_scores = torch.matmul(
            Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码（如填充掩码或未来信息掩码）
        if mask is not None:
            # mask 为 0 的位置替换为极小值，Softmax 后变为 0
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 对值向量加权求和
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # 将输入张量分割为多个头
        # 输入: (batch, seq_len, d_model) -> 输出: (batch, num_heads, seq_len, d_k)
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # 合并多个头
        # 输入: (batch, num_heads, seq_len, d_k) -> 输出: (batch, seq_len, d_model)
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # 线性变换并分头
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # 计算注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并头并输出变换
        output = self.W_o(self.combine_heads(attn_output))
        return output

# ==========================================
# 2. 基本组件：位置前馈网络 (Position-Wise Feed-Forward Network)
# ==========================================


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# ==========================================
# 3. 基本组件：位置编码 (Positional Encoding)
# ==========================================


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为 buffer，不是模型参数，不参与梯度更新
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # 将位置编码加到输入 Embedding 上
        return x + self.pe[:, :x.size(1)]

# ==========================================
# 4. 编码器层 (Encoder Layer)
# ==========================================


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 自注意力机制 + 残差连接 + 层归一化
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# ==========================================
# 5. 解码器层 (Decoder Layer)
# ==========================================


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 自注意力机制 (带掩码，防止看到未来信息)
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 交叉注意力机制 (Query来自解码器，Key/Value来自编码器)
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

# ==========================================
# 6. 完整的 Transformer 模型
# ==========================================


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout):
        super(Transformer, self).__init__()

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        # src_mask: 简单的填充掩码 (这里简化处理，假设没有 padding)
        # 实际应用中需要根据 padding token 也就是 0 来生成 mask
        src_mask = (src != 0).unsqueeze(
            1).unsqueeze(2)  # (batch, 1, 1, seq_len)

        # tgt_mask: 结合填充掩码和因果掩码 (Sequence Mask)
        tgt_mask = (tgt != 0).unsqueeze(
            1).unsqueeze(3)  # (batch, 1, seq_len, 1)
        seq_len = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        nopeak_mask = nopeak_mask.to(src.device)
        tgt_mask = tgt_mask & nopeak_mask  # (batch, 1, seq_len, seq_len)

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        # 生成掩码
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # 编码器部分
        src_emb = self.dropout(
            self.positional_encoding(self.encoder_embedding(src)))
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        # 解码器部分
        tgt_emb = self.dropout(
            self.positional_encoding(self.decoder_embedding(tgt)))
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)

        # 输出层
        output = self.fc(dec_output)
        return output

# ==========================================
# 7. 训练和评估
# ==========================================


def generate_random_batch(batch_size, src_len, vocab_size, device):
    """
    生成一个简单的复制任务数据：
    源输入: 随机序列 [a, b, c, ...]
    目标输出: 与源输入相同 [a, b, c, ...]
    模型需要学会将输入直接复制到输出
    """
    # 随机生成数据 (1 到 vocab_size-1，留出 0 作为 padding)
    data = torch.randint(1, vocab_size, (batch_size, src_len)).to(device)

    # 简单的复制任务：Src 和 Tgt 相同
    # 实际训练中，Decoder Input 需要有一个起始符 <sos> (这里用 0 代替演示)
    # Target Output 也是 data，模型学习预测下一个词

    # Src: [Batch, Seq_Len]
    src = data

    # Tgt: [Batch, Seq_Len + 1] -> <sos> + data
    # 构造 <sos> 列，这里为了简化，我们假设 0 是 <sos> 也是 padding
    # 在真实 NLP 任务中，需要专门的 <sos> token
    sos_token = torch.zeros((batch_size, 1), dtype=torch.long).to(device)
    tgt = torch.cat([sos_token, data], dim=1)

    return src, tgt


def train_epoch(model, criterion, optimizer, device):
    model.train()
    total_loss = 0

    # 模拟数据加载器 (这里使用随机数据进行演示)
    # 假设有 100 个 batch
    for _ in range(100):
        # 生成复制任务数据
        src, tgt = generate_random_batch(32, 10, src_vocab_size, device)

        # tgt_input: <sos> + seq (0 ... n-1)
        tgt_input = tgt[:, :-1]

        # tgt_output: seq + <eos> (这里简化为 seq)
        # 实际预测目标是下一个词
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()

        # 前向传播
        output = model(src, tgt_input)

        # output: (batch_size, tgt_len-1, tgt_vocab_size)
        # Reshape for loss calculation: (batch_size * (tgt_len-1), tgt_vocab_size)
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        loss.backward()

        # 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / 100


def evaluate(model, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for _ in range(10):  # 模拟 10 个验证 batch
            src, tgt = generate_random_batch(32, 10, src_vocab_size, device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)

            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            total_loss += loss.item()

    return total_loss / 10


def greedy_decode(model, src, max_len, start_symbol, device):
    """
    贪婪解码：每次选择概率最大的词作为下一个输入
    """
    model.eval()
    # src: (1, seq_len)
    # 初始化解码器输入: [[start_symbol]]
    tgt_input = torch.zeros((1, 1), dtype=torch.long).fill_(
        start_symbol).to(device)

    for i in range(max_len):
        # 前向传播，获取当前步的输出
        # 注意：这里我们简单地将当前生成的 tgt_input 作为输入
        # 实际 Transformer 需要 mask 防止看到未来，这里模型内部处理了
        with torch.no_grad():
            output = model(src, tgt_input)

        # 获取最后一个时间步的输出
        # output: (1, curr_seq_len, vocab_size)
        last_token_logits = output[:, -1, :]

        # 选择概率最大的索引
        _, next_token = torch.max(last_token_logits, dim=-1)
        next_token = next_token.item()

        # 将预测的 token 拼接到输入中，作为下一步的输入
        tgt_input = torch.cat([tgt_input, torch.ones(
            (1, 1), dtype=torch.long).to(device).fill_(next_token)], dim=1)

    # 去掉开头的 start_symbol
    return tgt_input[:, 1:]


if __name__ == "__main__":
    # 超参数设置
    # 缩小词表大小，让模型更容易学习复制任务
    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 256  # 减小模型维度加快训练
    num_heads = 4
    num_layers = 2  # 减少层数
    d_ff = 512
    max_len = 50
    dropout = 0.1
    epochs = 50  # 增加训练轮数

    # 实例化模型
    # 检查是否有 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model,
                        num_heads, num_layers, d_ff, max_len, dropout).to(device)

    # 初始化损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 假设 0 是 padding index
    optimizer = optim.Adam(model.parameters(), lr=0.0005,
                           betas=(0.9, 0.98), eps=1e-9)

    print("开始训练 (任务: 复制数字序列)...")
    print("-" * 50)

    for epoch in range(epochs):
        train_loss = train_epoch(model, criterion, optimizer, device)

        # 每 10 个 epoch 打印一次验证结果
        if (epoch + 1) % 10 == 0:
            val_loss = evaluate(model, criterion, device)
            print(
                f"Epoch: {epoch+1:03} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print("-" * 50)
    print("训练完成！")

    # 实际推理演示
    print("\n实际推理演示 (复制任务):")
    # 生成一个随机测试序列
    test_seq_len = 10
    src = torch.randint(1, src_vocab_size, (1, test_seq_len)).to(device)

    print(f"输入序列 (Source): {src[0].tolist()}")

    # 使用贪婪解码生成预测结果
    # 假设 0 是 <sos>
    pred_seq = greedy_decode(
        model, src, max_len=test_seq_len, start_symbol=0, device=device)

    print(f"预测序列 (Target): {pred_seq[0].tolist()}")

    # 简单对比
    if src[0].tolist() == pred_seq[0].tolist():
        print("\n✅ 成功！模型完美复制了输入序列。")
    else:
        print("\n⚠️ 部分不匹配，可能需要更多训练或调整参数。")
