import re
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"


"""
数据相关工具函数：
- 加载中英文平行语料
- 清洗文本、分词
- 构建词表并将句子转换为 ID 序列
- 对序列进行 PAD，得到固定长度的批次数据
"""


def load_custom_corpus(file_path):
    en_sentences = []
    zh_sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            en, zh = line.split("\t")
            en_sentences.append(en)
            zh_sentences.append(zh)
    return en_sentences, zh_sentences


en_corpus = [
    "Hello world.",
    "I love China.",
    "She likes reading books.",
    "This is a Transformer model.",
    "Machine learning is interesting.",
]

zh_corpus = [
    "你好，世界。",
    "我爱中国。",
    "她喜欢读书。",
    "这是一个Transformer模型。",
    "机器学习很有趣。",
]


def clean_text(text, is_zh=False):
    """
    文本清洗函数
    :param text: 原始句子
    :param is_zh: 是否为中文句子
    :return: 清洗后的文本字符串
    """
    if not is_zh:
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s\.]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
    else:
        text = re.sub(r"\s+", "", text).strip()
    return text


def tokenize_sentence(sent, is_zh=False):
    """
    对句子进行分词
    :param sent: 单个句子字符串
    :param is_zh: 中文则按字符切分，英文使用 nltk 分词
    :return: 分好词的列表
    """
    if is_zh:
        return list(sent)
    return nltk.word_tokenize(sent)


def build_vocab(token_lists):
    """
    根据分词结果构建词表
    :param token_lists: 多个句子的 token 序列列表
    :return: (token->id 字典, id->token 字典)
    """
    vocab = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}
    for tokens in token_lists:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)
    inv = {i: t for t, i in vocab.items()}
    return vocab, inv


def numericalize(token_lists, vocab):
    """
    将 token 序列转换为 ID 序列
    :param token_lists: 分词后的句子列表
    :param vocab: 词表映射 (token->id)
    :return: 对应的 ID 序列列表
    """
    return [[vocab[t] for t in tokens] for tokens in token_lists]


def pad_sequences(seqs, pad_idx):
    """
    对批次中的所有序列进行右侧 PAD
    :param seqs: 若干 ID 序列
    :param pad_idx: PAD 对应的 ID
    :return: (补齐后的序列列表, 最大长度)
    """
    max_len = max(len(s) for s in seqs)
    return [s + [pad_idx] * (max_len - len(s)) for s in seqs], max_len


def prepare_data():
    """
    准备训练所需的所有数据结构
    包括：
    - 清洗原始中英文语料
    - 分词并加上 SOS / EOS
    - 构建词表
    - 转为 ID 并做 PAD
    :return: 多个张量和词表，用于后续模型和训练
    """
    en_clean = [clean_text(sent) for sent in en_corpus]
    zh_clean = [clean_text(sent, is_zh=True) for sent in zh_corpus]

    en_tokens = [tokenize_sentence(sent) for sent in en_clean]
    zh_tokens = [tokenize_sentence(sent, is_zh=True) for sent in zh_clean]

    en_tokens = [[SOS_TOKEN] + tokens + [EOS_TOKEN] for tokens in en_tokens]
    zh_tokens = [[SOS_TOKEN] + tokens + [EOS_TOKEN] for tokens in zh_tokens]

    en_vocab, en_inv = build_vocab(en_tokens)
    zh_vocab, zh_inv = build_vocab(zh_tokens)

    en_ids = numericalize(en_tokens, en_vocab)
    zh_ids = numericalize(zh_tokens, zh_vocab)

    en_padded, en_maxlen = pad_sequences(en_ids, en_vocab[PAD_TOKEN])
    zh_padded, zh_maxlen = pad_sequences(zh_ids, zh_vocab[PAD_TOKEN])

    return (
        en_padded,
        zh_padded,
        en_vocab,
        zh_vocab,
        en_inv,
        zh_inv,
        en_maxlen,
        zh_maxlen,
    )
