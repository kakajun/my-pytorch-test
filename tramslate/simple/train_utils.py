import torch
from torch.utils.data import DataLoader

from .model import create_padding_mask


"""
训练与评估相关工具：
- collate: 将原始样本打包成模型输入需要的张量
- create_dataloaders: 构建训练集和验证集的 DataLoader
- train_one_epoch: 训练一个 epoch，返回平均损失
- evaluate: 在验证集上评估一次，返回损失
"""


def collate(batch, device, src_pad_idx, tgt_pad_idx):
    """
    将一批样本转换为张量形式
    :param batch: [(src_ids, tgt_ids), ...]
    :param device: 运行设备
    :param src_pad_idx: 源语言 PAD ID（这里只是保持接口一致）
    :param tgt_pad_idx: 目标语言 PAD ID（这里只是保持接口一致）
    :return: (src_batch, tgt_in, tgt_out)
    """
    src_batch = torch.tensor([b[0] for b in batch], dtype=torch.long)
    tgt_batch = torch.tensor([b[1] for b in batch], dtype=torch.long)
    src_batch = src_batch.transpose(0, 1).to(device)
    tgt_batch = tgt_batch.transpose(0, 1).to(device)
    tgt_in = tgt_batch[:-1, :]
    tgt_out = tgt_batch[1:, :]
    return src_batch, tgt_in, tgt_out


def create_dataloaders(en_padded, zh_padded, batch_size, device, src_pad_idx, tgt_pad_idx):
    """
    构建训练集和验证集的 DataLoader
    简单起见：最后一条样本作为验证集，其余作为训练集
    """
    data = list(zip(en_padded, zh_padded))
    train_data = data[:-1]
    val_data = data[-1:]

    def collate_fn(batch):
        return collate(batch, device, src_pad_idx, tgt_pad_idx)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_data, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, val_loader


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    src_pad_idx,
    tgt_pad_idx,
):
    """
    训练一个 epoch
    :param model: Transformer 模型
    :param train_loader: 训练集 DataLoader
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param src_pad_idx: 源 PAD ID，用于构建 mask
    :param tgt_pad_idx: 目标 PAD ID，用于构建 mask
    :return: 当前 epoch 的平均 loss
    """
    model.train()
    total = 0.0
    for src, tgt_in, tgt_out in train_loader:
        src_mask = create_padding_mask(src, src_pad_idx)
        tgt_mask = create_padding_mask(tgt_in, tgt_pad_idx)
        logits = model(
            src,
            tgt_in,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask,
        )
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(train_loader))


def evaluate(
    model,
    val_loader,
    criterion,
    src_pad_idx,
):
    """
    在验证集上评估模型
    :param model: 已训练模型
    :param val_loader: 验证集 DataLoader
    :param criterion: 损失函数
    :param src_pad_idx: 源 PAD ID，用于构建 mask
    :return: 验证集 loss
    """
    model.eval()
    with torch.no_grad():
        for src, tgt_in, tgt_out in val_loader:
            src_mask = create_padding_mask(src, src_pad_idx)
            logits = model(
                src,
                tgt_in,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=src_mask,
            )
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.reshape(-1))
            return loss.item()
