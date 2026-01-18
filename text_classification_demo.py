import random
import time
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import os
# 解决 OpenMP 重复初始化错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ==========================================
# 0. 依赖检查与设置
# ==========================================

# 检查 jieba 分词库
try:
    import jieba
except ImportError:
    print("\n" + "="*50)
    print("错误：未找到 jieba 分词库。")
    print("请运行: pip install jieba")
    print("="*50 + "\n")
    raise

# 设置随机种子以保证结果可复现
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 尝试导入 torchtext
try:
    from torchtext.legacy.data import Field, TabularDataset, BucketIterator
    print("Using torchtext.legacy.data")
except ImportError:
    try:
        from torchtext.data import Field, TabularDataset, BucketIterator
        print("Using torchtext.data")
    except ImportError:
        print("\n" + "="*50)
        print("错误：无法从 torchtext 导入 Field, TabularDataset, BucketIterator。")
        print("建议安装旧版本 torchtext: pip install torchtext==0.6.0")
        print("="*50 + "\n")
        raise

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ==========================================
# 1. 数据准备 (生成中文模拟数据)
# ==========================================
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

TRAIN_PATH = os.path.join(DATA_DIR, 'train_cn.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test_cn.csv')


def generate_chinese_dummy_data(path, num_samples=100):
    data = {
        'text': [],
        'label': []
    }

    # 简单的词库
    pos_words = ["喜欢", "棒", "精彩", "优秀", "好", "爱", "推荐", "开心", "完美", "感动"]
    neg_words = ["讨厌", "差", "糟糕", "无聊", "烂", "坏", "失望", "难看", "垃圾", "浪费"]
    neutral_words = ["电影", "这", "个", "是", "剧情",
                     "演员", "真的", "很", "了", "看", "觉得", "时间"]

    for _ in range(num_samples):
        length = random.randint(5, 20)
        sentence = []

        # 随机决定生成正面还是负面样本
        is_positive = random.random() > 0.5

        # 构建句子
        for _ in range(length):
            r = random.random()
            if r < 0.4:  # 40% 概率插入情感词
                if is_positive:
                    word = random.choice(pos_words)
                else:
                    word = random.choice(neg_words)
            else:  # 60% 概率插入中性词
                word = random.choice(neutral_words)
            sentence.append(word)

        # 简单计算分数来确定标签（双重保险）
        score = 0
        for word in sentence:
            if word in pos_words:
                score += 1
            if word in neg_words:
                score -= 1

        # 如果分数中性，强制加一个情感词
        if score == 0:
            if is_positive:
                sentence.append(random.choice(pos_words))
                score += 1
            else:
                sentence.append(random.choice(neg_words))
                score -= 1

        label = 1 if score > 0 else 0
        data['text'].append("".join(sentence))  # 中文不加空格连接
        data['label'].append(label)

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"已生成中文模拟数据: {path}")


# 无论是否存在，重新生成以确保是中文数据 (或者检查文件内容，这里简单起见如果用户要求中文就重新生成)
print("正在检查/生成数据...")
generate_chinese_dummy_data(TRAIN_PATH, num_samples=1000)
generate_chinese_dummy_data(TEST_PATH, num_samples=200)

# ==========================================
# 2. 定义字段和加载数据
# ==========================================
print("Loading data...")

# 中文分词函数


def chinese_tokenizer(text):
    return [w for w in jieba.lcut(text) if w.strip()]  # 去除空格


# 定义字段处理
TEXT = Field(tokenize=chinese_tokenizer,
             include_lengths=True)

LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float)

# 加载数据集
train_data, test_data = TabularDataset.splits(
    path=DATA_DIR,
    train='train_cn.csv',
    test='test_cn.csv',
    format='csv',
    skip_header=True,
    fields=[('text', TEXT), ('label', LABEL)]
)

print(f'训练集数量: {len(train_data)}')
print(f'测试集数量: {len(test_data)}')

# 构建词汇表
# 中文 demo 暂时不使用预训练词向量 (GloVe 是英文的)，使用随机初始化
print("Building vocabulary...")
TEXT.build_vocab(train_data, max_size=25000)

print(f"词汇表大小: {len(TEXT.vocab)}")
print(f"常见词汇: {TEXT.vocab.freqs.most_common(10)}")

# 创建迭代器
BATCH_SIZE = 64

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.text),
    device=device)

# ==========================================
# 3. 模型构建 (LSTM)
# ==========================================


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.to('cpu'))
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # 拼接双向 LSTM 的最后一个 hidden state
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return self.fc(hidden)


# ==========================================
# 4. 训练设置
# ==========================================
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = SentimentLSTM(INPUT_DIM,
                      EMBEDDING_DIM,
                      HIDDEN_DIM,
                      OUTPUT_DIM,
                      N_LAYERS,
                      BIDIRECTIONAL,
                      DROPOUT,
                      PAD_IDX)

model = model.to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# ==========================================
# 5. 开始训练
# ==========================================
N_EPOCHS = 5
print(f"开始训练，共 {N_EPOCHS} 个 Epoch...")

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_iterator, criterion)

    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

    print(f'Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

# ==========================================
# 6. 模型应用 (中文预测)
# ==========================================


def predict_sentiment(model, sentence):
    model.eval()
    # 使用 jieba 分词
    tokenized = [w for w in jieba.lcut(sentence) if w.strip()]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    if length[0] == 0:  # 处理空句子
        return 0.5

    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)  # [sent len, 1]
    length_tensor = torch.LongTensor(length)

    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()


print("\n=== 中文情感分析演示 ===")
pos_review = "这部电影真的很棒，剧情精彩，演员演技在线，强烈推荐！"
neg_review = "太失望了，剧情无聊，浪费了我的时间，非常糟糕。"
neutral_review = "这只是一部普通的电影，不好也不坏。"

print(f"句子: '{pos_review}'")
print(f"情感得分: {predict_sentiment(model, pos_review):.4f} (接近 1 为正面)")

print(f"句子: '{neg_review}'")
print(f"情感得分: {predict_sentiment(model, neg_review):.4f} (接近 0 为负面)")

print(f"句子: '{neutral_review}'")
print(f"情感得分: {predict_sentiment(model, neutral_review):.4f}")
