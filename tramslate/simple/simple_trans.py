import os
import sys

import torch
import torch.nn as nn

# 兼容两种运行方式：
# 1) 作为脚本直接运行：python tramslate/simple/simple_trans.py
# 2) 作为模块运行：python -m tramslate.simple.simple_trans
if __package__ is None or __package__ == "":
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )

from tramslate.simple.data import (
    prepare_data,
    SOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
)
from tramslate.simple.model import (
    TransformerSeq2Seq,
    greedy_decode,
)
from tramslate.simple.train_utils import (
    create_dataloaders,
    train_one_epoch,
    evaluate,
)


def main():
    """
    整个翻译 demo 的入口函数：
    1. 准备数据和词表
    2. 构建 DataLoader
    3. 创建 Transformer 模型、损失函数和优化器
    4. 训练若干轮并打印损失
    5. 使用贪心解码对一个示例句子进行翻译
    """
    device = torch.device("cpu")

    (
        en_padded,
        zh_padded,
        en_vocab,
        zh_vocab,
        en_inv,
        zh_inv,
        en_maxlen,
        zh_maxlen,
    ) = prepare_data()

    src_pad_idx = en_vocab[PAD_TOKEN]
    tgt_pad_idx = zh_vocab[PAD_TOKEN]

    train_loader, val_loader = create_dataloaders(
        en_padded,
        zh_padded,
        batch_size=2,
        device=device,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
    )

    model = TransformerSeq2Seq(len(en_vocab), len(zh_vocab)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        tr = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            src_pad_idx,
            tgt_pad_idx,
        )
        vl = evaluate(
            model,
            val_loader,
            criterion,
            src_pad_idx,
        )
        print(f"epoch={epoch+1} train_loss={tr:.4f} val_loss={vl:.4f}")

    test_src = en_padded[0]
    decoded = greedy_decode(
        model,
        test_src,
        zh_maxlen,
        src_pad_idx,
        zh_vocab[SOS_TOKEN],
        zh_vocab[EOS_TOKEN],
        zh_inv,
        device,
    )
    src_tokens = [en_inv[i] for i in test_src if i != src_pad_idx]
    print("src:", src_tokens)
    print("pred:", decoded)


if __name__ == "__main__":
    main()
