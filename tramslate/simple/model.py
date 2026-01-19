import math
import torch
import torch.nn as nn


"""
模型定义模块：
- PositionalEncoding: 位置编码，给序列中每个位置加入位置信息
- TransformerSeq2Seq: 基于 nn.Transformer 的简单中英翻译模型
- create_padding_mask / greedy_decode: 辅助的 Mask 与解码函数
"""


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        构建固定的正弦 / 余弦位置编码矩阵
        :param d_model: 特征维度
        :param max_len: 支持的最大序列长度
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        """
        将前 seq_len 行的位置编码加到输入 x 上
        :param x: [seq_len, batch_size, d_model]
        :return: 加上位置编码后的张量
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len]


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        """
        简单的 Transformer 编码器-解码器结构
        :param src_vocab_size: 源语言词表大小
        :param tgt_vocab_size: 目标语言词表大小
        :param d_model: 词向量维度
        :param nhead: 多头注意力头数
        :param num_encoder_layers: 编码器层数
        :param num_decoder_layers: 解码器层数
        :param dim_feedforward: 前馈层维度
        :param dropout: dropout 概率
        """
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src,
        tgt_in,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        前向传播
        :param src: 源序列 ID，[src_len, batch_size]
        :param tgt_in: 目标序列输入端（带 SOS），[tgt_len, batch_size]
        :param src_key_padding_mask: 源序列 PAD mask
        :param tgt_key_padding_mask: 目标序列 PAD mask
        :param memory_key_padding_mask: 编码器输出对应的 mask
        :return: 每个时间步的分类 logits，[tgt_len, batch_size, tgt_vocab_size]
        """
        src = self.src_embed(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embed(tgt_in) * math.sqrt(self.d_model)
        src = self.pos_enc(src)
        tgt = self.pos_enc(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(
            src.device
        )
        out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.generator(out)
        return logits


def create_padding_mask(batch_seqs, pad_idx):
    """
    根据 PAD ID 构建 key_padding_mask
    :param batch_seqs: [seq_len, batch_size] 的 ID 序列
    :param pad_idx: PAD 对应的 ID
    :return: [batch_size, seq_len] 的 bool mask，PAD 位置为 True
    """
    return (batch_seqs == pad_idx).transpose(0, 1)


def greedy_decode(
    model,
    src_tokens,
    max_len,
    src_pad_idx,
    sos_idx,
    eos_idx,
    id2token,
    device,
):
    """
    使用贪心策略进行解码
    每一步都选择概率最大的下一个 token，直到生成 EOS 或达到最大长度
    :param model: 训练好的 Transformer 模型
    :param src_tokens: 单个源句子的 ID 序列
    :param max_len: 解码的最大步数
    :param src_pad_idx: 源语言 PAD ID
    :param sos_idx: 目标语言 SOS ID
    :param eos_idx: 目标语言 EOS ID
    :param id2token: 目标语言 id->token 字典
    :param device: 运算设备
    :return: 生成的 token 序列（字符串列表）
    """
    model.eval()
    src = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(1).to(device)
    src_mask = create_padding_mask(src, src_pad_idx)
    ys = torch.tensor([sos_idx], dtype=torch.long).unsqueeze(1).to(device)
    for _ in range(max_len):
        tgt_in = ys
        logits = model(
            src,
            tgt_in,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=src_mask,
        )
        next_token = logits[-1].argmax(-1)
        ys = torch.cat([ys, next_token.unsqueeze(0)], dim=0)
        if next_token.item() == eos_idx:
            break
    tokens = [id2token[t.item()] for t in ys.squeeze(1)]
    return tokens
