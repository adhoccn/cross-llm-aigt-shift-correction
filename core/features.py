# -*- coding: utf-8 -*-
"""
15D 基线特征与 10D 扩展特征（对应论文表4 / 表5）。

word_count / avg_word_length / unique_word_ratio 在中文上依赖分词口径，
本模块提供两种分词器：
    tokenizer='jieba'  词级分词（默认）
    tokenizer='char'   字符级
"""
import re
import math
from collections import Counter

SENT_SPLIT = re.compile(r'[。！？!?；;…]+|\n+')
PARA_SPLIT = re.compile(r'\n+')
CJK = re.compile(r'[\u4e00-\u9fff]')
LATIN_WORD = re.compile(r'[A-Za-z]+')
DIGIT = re.compile(r'[0-9０-９]')
PUNCT_ALL = re.compile(r'[，。！？；：、“”‘’（）《》〈〉【】…—\-—,.!?;:()"\']')
COMMA = re.compile(r'[，,、]')
ENDING = re.compile(r'[。！？!?…]')

# 功能词表：论文未给出，采用通用高频功能词
FUNCTION_WORDS = set(
    '的 了 和 是 在 我 有 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这 '
    '那 我们 你们 他们 它 之 与 及 或 而 且 但 因为 所以 如果 可以 这个 那个 什么 怎么 为什么 '
    '对于 关于 通过 根据 由于 因此 于是 并且 或者 以及 中的 对于 来说 一些 这些 那些'.split()
)

POS_WORDS = set('好 喜欢 优秀 满意 开心 高兴 成功 有效 方便 简单 推荐 值得 美好 帮助 改善 提升'.split())
NEG_WORDS = set('差 坏 讨厌 失望 难过 失败 无效 麻烦 复杂 不推荐 糟糕 严重 困难 问题 风险 损失'.split())


def _tokens(text, tokenizer='jieba'):
    if tokenizer == 'jieba':
        try:
            import jieba
            return [t for t in jieba.lcut(text) if t.strip()]
        except Exception:
            tokenizer = 'char'
    # char 模式：英文数字串为词，汉字逐字为词
    return re.findall(r'[A-Za-z0-9]+|[\u4e00-\u9fff]', text)


def _sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def _paragraphs(text):
    return [p.strip() for p in PARA_SPLIT.split(text) if p.strip()]


def _entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values())


def _syntax_depth(text):
    """句法树深度近似：括号/引号嵌套的最大深度（论文 §3.2.2 说明其由嵌套结构决定）"""
    pairs = {'(': ')', '（': '）', '[': ']', '【': '】', '{': '}', '《': '》', '"': '"', "'": "'"}
    depth = mx = 0
    stack = []
    for ch in text:
        if ch in pairs:
            stack.append(ch); depth += 1; mx = max(mx, depth)
        elif stack and ch == pairs.get(stack[-1]):
            stack.pop(); depth -= 1
    # 逗号密度作为次级贡献
    return mx + 0.5 * len(COMMA.findall(text)) / max(1, len(text) / 50)


def extract_15d(text, tokenizer='jieba'):
    """返回 (dict, ) 15 维基线特征"""
    if not isinstance(text, str) or not text.strip():
        return {k: 0.0 for k in FEATURES_15D}
    toks = _tokens(text, tokenizer)
    sents = _sentences(text)
    paras = _paragraphs(text)
    n_char = len(text)

    sent_lens = [len(s) for s in sents] or [0]
    para_lens = [len(p) for p in paras] or [0]

    n_tok = len(toks)
    uniq_tok = len(set(toks))
    avg_word_len = (sum(len(t) for t in toks) / n_tok) if n_tok else 0.0

    n_punct = len(PUNCT_ALL.findall(text))
    n_comma = len(COMMA.findall(text))
    ends = [s[-1] for s in sents if s and ENDING.match(s[-1])]
    end_div = (len(set(ends)) / len(ends)) if ends else 0.0

    func_cnt = sum(1 for t in toks if t in FUNCTION_WORDS)
    n_digit = len(DIGIT.findall(text))
    n_upper = len(re.findall(r'[A-Z]', text))
    uniq_char = len(set(CJK.findall(text)))
    n_cjk = len(CJK.findall(text))

    return {
        'word_count': float(n_tok),
        'unique_word_ratio': (uniq_tok / n_tok) if n_tok else 0.0,
        'avg_word_length': avg_word_len,
        'sentence_count': float(len(sents)),
        'avg_sentence_length': float(sum(sent_lens) / len(sent_lens)),
        'sentence_length_std': float(_std(sent_lens)),
        'paragraph_count': float(len(paras)),
        'avg_paragraph_length': float(sum(para_lens) / len(para_lens)),
        'punctuation_ratio': (n_punct / n_char) if n_char else 0.0,
        'comma_ratio': (n_comma / n_char) if n_char else 0.0,
        'ending_punctuation_diversity': end_div,
        'function_word_ratio': (func_cnt / n_tok) if n_tok else 0.0,
        'character_diversity': (uniq_char / n_cjk) if n_cjk else 0.0,
        'digit_ratio': (n_digit / n_char) if n_char else 0.0,
        'uppercase_ratio': (n_upper / n_char) if n_char else 0.0,
    }


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))  # 总体 std


def extract_10d(text, tokenizer='jieba'):
    """10 维扩展特征（含外部依赖，见论文表5）"""
    if not isinstance(text, str) or not text.strip():
        return {k: 0.0 for k in FEATURES_10D}
    toks = _tokens(text, tokenizer)
    sents = _sentences(text)
    n_tok = len(toks)

    char_ent = _entropy(Counter(text))
    word_ent = _entropy(Counter(toks))

    # Zipf slope: log(rank) ~ log(freq) 线性回归斜率
    freqs = sorted(Counter(toks).values(), reverse=True)
    zipf = 0.0
    if len(freqs) >= 3:
        xs = [math.log(i + 1) for i in range(len(freqs))]
        ys = [math.log(f) for f in freqs]
        zipf = _slope(xs, ys)

    pos = neg = neu = 0
    for t in toks:
        if t in POS_WORDS:
            pos += 1
        elif t in NEG_WORDS:
            neg += 1
        else:
            neu += 1
    total_sent = max(1, n_tok)
    # 逐句情感极性方差
    pol = []
    for s in sents:
        st = _tokens(s, tokenizer)
        p = sum(1 for t in st if t in POS_WORDS)
        n = sum(1 for t in st if t in NEG_WORDS)
        pol.append((p - n) / max(1, p + n + 1))
    sent_var = _std(pol) ** 2

    func_types = len({t for t in toks if t in FUNCTION_WORDS})
    func_total = sum(1 for t in toks if t in FUNCTION_WORDS)
    func_div = (func_types / func_total) if func_total else 0.0

    # 相邻句 Jaccard 相似度均值
    sims = []
    for a, b in zip(sents, sents[1:]):
        sa, sb = set(_tokens(a, tokenizer)), set(_tokens(b, tokenizer))
        sims.append(len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0)

    return {
        'char_entropy': char_ent,
        'word_entropy': word_ent,
        'zipf_slope': zipf,
        'sentiment_positive_ratio': pos / total_sent,
        'sentiment_negative_ratio': neg / total_sent,
        'sentiment_neutral_ratio': neu / total_sent,
        'sentiment_variance': sent_var,
        'function_word_diversity': func_div,
        'inter_sentence_similarity': (sum(sims) / len(sims)) if sims else 0.0,
        'syntax_tree_depth': _syntax_depth(text),
    }


def _slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


FEATURES_15D = [
    'word_count', 'unique_word_ratio', 'avg_word_length',
    'sentence_count', 'avg_sentence_length', 'sentence_length_std',
    'paragraph_count', 'avg_paragraph_length',
    'punctuation_ratio', 'comma_ratio', 'ending_punctuation_diversity',
    'function_word_ratio', 'character_diversity', 'digit_ratio', 'uppercase_ratio',
]

FEATURES_10D = [
    'char_entropy', 'word_entropy', 'zipf_slope',
    'sentiment_positive_ratio', 'sentiment_negative_ratio', 'sentiment_neutral_ratio',
    'sentiment_variance', 'function_word_diversity',
    'inter_sentence_similarity', 'syntax_tree_depth',
]


def extract(text, dim=15, tokenizer='jieba'):
    d = extract_15d(text, tokenizer)
    if dim == 25:
        d.update(extract_10d(text, tokenizer))
    return d


def extract_matrix(texts, dim=15, tokenizer='jieba'):
    import numpy as np
    keys = FEATURES_15D if dim == 15 else FEATURES_15D + FEATURES_10D
    out = np.zeros((len(texts), len(keys)), dtype=np.float64)
    for i, t in enumerate(texts):
        d = extract(t, dim=dim, tokenizer=tokenizer)
        for j, k in enumerate(keys):
            out[i, j] = d.get(k, 0.0)
    return out, keys
