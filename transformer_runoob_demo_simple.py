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
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
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
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2) # (batch, 1, 1, seq_len)
        
        # tgt_mask: 结合填充掩码和因果掩码 (Sequence Mask)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3) # (batch, 1, seq_len, 1)
        seq_len = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        nopeak_mask = nopeak_mask.to(src.device)
        tgt_mask = tgt_mask & nopeak_mask # (batch, 1, seq_len, seq_len)
        
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        # 生成掩码
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # 编码器部分
        src_emb = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
            
        # 解码器部分
        tgt_emb = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
            
        # 输出层
        output = self.fc(dec_output)
        return output

# ==========================================
# 7. 运行演示
# ==========================================
if __name__ == "__main__":
    # 超参数设置
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_len = 100
    dropout = 0.1
    
    # 实例化模型
    # 检查是否有 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout).to(device)
    
    # 模拟输入数据 (Batch Size=32, Sequence Length=10)
    # 注意：这里我们随机生成数据，不包含 padding (0)，所以 mask 全为 True
    src = torch.randint(1, src_vocab_size, (32, 10)).to(device)
    tgt = torch.randint(1, tgt_vocab_size, (32, 20)).to(device)
    
    # 前向传播
    output = model(src, tgt)
    
    print("\n模型结构:")
    # print(model) # 打印模型结构太长，注释掉
    print("Transformer model created successfully!")
    
    print("\n输入尺寸:")
    print(f"Source: {src.shape}")
    print(f"Target: {tgt.shape}")
    
    print("\n输出尺寸:")
    print(f"Output: {output.shape}") # 预期: (32, 20, tgt_vocab_size)
    
    print("\n验证成功！这是一个完整的 Transformer 模型演示。")
