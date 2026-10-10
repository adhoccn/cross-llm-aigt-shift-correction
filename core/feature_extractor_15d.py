"""AI文本检测 — 15D特征提取器（独立模块，无实验框架依赖，无外部NLP库）

在10D语言学特征基础上，新增5个基于规则的句法特征，无需安装spaCy/stanza。

用法：
    from core.feature_extractor_15d import extract_features_15d
    features = extract_features_15d("这是一段文本...")
    # features: numpy array of shape (15,)
    # 0-9: 10 个语言学特征
    # 10-14: 5 个句法特征
"""
import numpy as np
import re, math
from collections import Counter

# 编译一次的正则
_SENTENCE_RE = re.compile(r'[。！？.!?；\n]+')

# 常量集合
_PUNCT_SET = frozenset('，。！？、；：""''【】（）…—·,\'"!?;:()[]{}-')
_FUNC_WORDS = frozenset('的了是在我你他她它们这那个一不有和与而但如果因为所以被把让给从向到对也已还又很都就才能吗呢吧啊呀哇嘛呗哦哈喽耶唔')
_FUNC_WORDS_BURST = frozenset('的了是在我你他她它们这那个一不有和与而但如果因为所以被把让给从向到对也已还又很都就才能')
_COLLOQUIAL_MARKERS = ['吧','啊','呢','嘛','哦','哈','嗯','哇','呀','诶',
                       '说白了','换句话说','举个例子','对吧','是不是',
                       '反正','总之','等等','之类的','什么的','大概','差不多',
                       '挺','蛮','超','巨']
_IDIOMS = ['不言而喻','众所周知','与此同时','总而言之','综上所述',
           '前所未有','不可忽视','息息相关','举足轻重','层出不穷',
           '日益增长','蓬勃发展','广泛关注','深入探讨','有机结合',
           '一方面','另一方面','换句话说','总的来说','具体来说']

# 句法相关常量
_CLAUSE_MARKERS = ['如果','要是','假如','倘若','若','即使','即便','尽管','虽然','虽说',
                   '因为','由于','因此','因而','从而','所以','于是','结果','那么',
                   '但是','但','然而','不过','只是','却','可是','而','而且','并且','况且',
                   '当','在','随着','自从','直到','等到','一...就','只要','只有','除非',
                   '无论','不管','不论','不论','无论','无论','不管','不论']
_PASSIVE_MARKERS = ['被','让','给','受','遭','由','为...所','见...于']
_MODIFIER_PARTICLES = ['的','地','得']

# 工具函数
def _safe_mean(data):
    if not data: return 0.0
    return sum(data) / len(data)

def _safe_var(data):
    if len(data) < 2: return 0.0
    mean = sum(data) / len(data)
    return sum((x - mean) ** 2 for x in data) / len(data)

def _safe_std(data):
    return math.sqrt(_safe_var(data))


def _preprocess(text):
    """统一预处理：一次提取所有中间结果"""
    chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
    n_chars = len(chars)

    raw_sentences = _SENTENCE_RE.split(text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 3]
    sentence_lengths = [len(s) for s in sentences]

    raw_paragraphs = text.split('\n')
    paragraphs = [p.strip() for p in raw_paragraphs if len(p.strip()) > 10]
    paragraph_lengths = [len(p) for p in paragraphs]

    if len(paragraphs) < 2 and len(sentences) >= 4:
        seg = max(len(sentences) // 3, 1)
        paragraphs = [''.join(sentences[i:i+seg]) for i in range(0, len(sentences), seg)]
        paragraph_lengths = [len(p) for p in paragraphs]

    char_counts = Counter(chars)

    return {
        'text': text, 'chars': chars, 'n_chars': n_chars,
        'sentences': sentences, 'sentence_lengths': sentence_lengths,
        'paragraphs': paragraphs, 'paragraph_lengths': paragraph_lengths,
        'char_counts': char_counts,
    }


def _compute_baseline_features(p):
    """计算10D基线特征"""
    text = p['text']
    c = p['chars']
    n = p['n_chars']
    ss = p['sentences']
    sl = p['sentence_lengths']
    pl = p['paragraph_lengths']
    cc = p['char_counts']

    # 1. 句长方差
    f0 = _safe_var(sl)

    # 2. 重复率
    if len(c) < 10:
        f1 = 0.0
    else:
        tg = [tuple(c[i:i+3]) for i in range(len(c)-2)]
        tc = Counter(tg)
        f1 = sum(v-1 for v in tc.values() if v > 1) / len(tg)

    # 3. 突发性
    if len(c) < 20:
        f2 = 0.0
    else:
        seg_len = len(c) // 4
        segs = [c[i*seg_len:(i+1)*seg_len] for i in range(4)]
        wc = Counter(ch for ch in c if ch not in _FUNC_WORDS_BURST)
        top = [w for w, _ in wc.most_common(10)]
        if not top:
            f2 = 0.0
        else:
            variances = []
            for ch in top:
                counts = [seg.count(ch) / max(len(seg), 1) for seg in segs]
                mean_c = _safe_mean(counts)
                var_c = sum((x - mean_c) ** 2 for x in counts) / len(counts)
                variances.append(var_c)
            f2 = sum(variances) / len(variances)

    # 4. 功能词比
    f3 = sum(1 for ch in c if ch in _FUNC_WORDS) / max(n, 1)

    # 5. 标点特征
    punct_set = set(ch for ch in text if ch in _PUNCT_SET)
    diversity = len(punct_set) / len(_PUNCT_SET)
    excl = (text.count('！') + text.count('!')) / max(n, 1) * 10
    ellipsis = (text.count('…') + text.count('......')) / max(n, 1) * 5
    if len(sl) < 2:
        f4 = diversity
    else:
        mean_sl = _safe_mean(sl)
        std_sl = _safe_std(sl)
        cv = std_sl / (mean_sl + 1e-6)
        f4 = diversity + excl + ellipsis + cv * 0.1

    # 6. 字频熵
    if n < 20:
        f5 = 0.0
    else:
        total = float(n)
        ps_list = [cnt / total for cnt in cc.values()]
        f5 = -sum(p_ * math.log2(p_) for p_ in ps_list)

    # 7. 段长方差
    f6 = _safe_var(pl)

    # 8. 口语标记
    f7 = sum(text.count(m) for m in _COLLOQUIAL_MARKERS) / max(n, 1) * 100

    # 9. 成语密度
    idiom_count = 0
    for i in _IDIOMS:
        idiom_count += text.count(i)
    f8 = idiom_count / max(n, 1) * 100

    # 10. 首尾连贯
    if len(ss) < 4:
        f9 = 0.0
    else:
        f_set = set(ss[0])
        l_set = set(ss[-1])
        o1 = len(f_set & l_set) / min(len(f_set), len(l_set)) if f_set and l_set else 0.0
        f3_set = set(''.join(ss[:3]))
        l3_set = set(''.join(ss[-3:]))
        o3 = len(f3_set & l3_set) / min(len(f3_set), len(l3_set)) if f3_set and l3_set else 0.0
        f9 = o1 * 0.5 + o3 * 0.5

    return f0, f1, f2, f3, f4, f5, f6, f7, f8, f9


def _compute_syntax_features(text, sentences):
    """计算5D句法特征（基于规则，无需外部NLP库）
    
    11. 句法嵌套深度: 基于逗号分层+括号/引号嵌套
    12. 从句密度: 从句标记数/句子数
    13. 被动句比例: 被动标记数/句子数
    14. 修饰语密度: 的/地/得数量/总字数
    15. 句法多样性: 句子开头模式熵
    """
    if not sentences:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    n = max(sum(1 for c in text if '\u4e00' <= c <= '\u9fff'), 1)
    n_sent = len(sentences)
    
    # 11. 句法嵌套深度（规则近似）
    # 每句的逗号/顿号密度 + 引号/括号嵌套层数
    depths = []
    for s in sentences:
        # 计算该句中的嵌套层数
        nest = 0
        max_nest = 0
        for ch in s:
            if ch in '（(【[《""''『':
                nest += 1
                max_nest = max(max_nest, nest)
            elif ch in '）)】]》""''』':
                nest = max(0, nest - 1)
        # 逗号/顿号密度作为额外信号
        comma_density = (s.count('，') + s.count('、')) / max(len(s), 1)
        depth = max_nest + comma_density * 2
        depths.append(depth)
    f10 = _safe_mean(depths)
    
    # 12. 从句密度
    clause_count = 0
    for marker in _CLAUSE_MARKERS:
        # 简单统计，不精确匹配"为...所"等复杂模式
        if '...' not in marker:
            clause_count += text.count(marker)
    f11 = clause_count / max(n_sent, 1)
    
    # 13. 被动句比例
    passive_count = 0
    for s in sentences:
        # 被动标记：被/让/给/受/遭
        for m in ['被','让','给','受','遭']:
            if m in s:
                passive_count += 1
                break  # 每句只计一次
    f12 = passive_count / max(n_sent, 1)
    
    # 14. 修饰语密度（的/地/得）
    mod_count = sum(text.count(p) for p in _MODIFIER_PARTICLES)
    f13 = mod_count / max(n, 1) * 100
    
    # 15. 句法多样性（句子开头模式熵）
    # 提取每句前2-4个汉字作为"开头模式"
    patterns = []
    for s in sentences:
        # 取前2-4个汉字
        s_clean = re.sub(r'[^\u4e00-\u9fff]', '', s)
        if len(s_clean) >= 2:
            pattern = s_clean[:min(4, len(s_clean))]
            patterns.append(pattern)
    
    if patterns:
        pc = Counter(patterns)
        total = len(patterns)
        f14 = -sum((cnt/total) * math.log2(cnt/total) for cnt in pc.values())
    else:
        f14 = 0.0
    
    return f10, f11, f12, f13, f14


def extract_features_15d(text):
    """提取15D特征向量（10D语言学 + 5D句法）
    
    Args:
        text: str, 输入文本（建议>50字）
    
    Returns:
        numpy.ndarray, shape=(15,), dtype=float64
    
    特征顺序:
        0: 句长方差
        1: 重复率
        2: 突发性
        3: 功能词比
        4: 标点特征
        5: 字频熵
        6: 段长方差
        7: 口语标记
        8: 成语密度
        9: 首尾连贯
        10: 句法嵌套深度
        11: 从句密度
        12: 被动句比例
        13: 修饰语密度
        14: 句法多样性
    """
    p = _preprocess(text)
    baseline = _compute_baseline_features(p)
    syntax = _compute_syntax_features(p['text'], p['sentences'])
    return np.array(baseline + syntax, dtype=np.float64)


def extract_features_10d(text):
    """向后兼容：提取10D特征向量"""
    p = _preprocess(text)
    feats = _compute_baseline_features(p)
    return np.array(feats, dtype=np.float64)


def extract_features_batch(texts, dim=15):
    """批量提取特征
    
    Args:
        texts: List[str]
        dim: 15 或 10
    
    Returns:
        numpy.ndarray, shape=(n, dim)
    """
    extractor = extract_features_15d if dim == 15 else extract_features_10d
    return np.array([extractor(t) for t in texts], dtype=np.float64)


if __name__ == '__main__':
    # 测试对比
    human_text = "今天天气真好，我们一起去公园玩吧。公园里有很多人在散步，还有一些孩子在放风筝。\
总的来说，这是一个非常愉快的周末。你觉得呢？"
    
    ai_text = "人工智能正在改变世界。它可以帮助我们解决许多复杂的问题，提高生产效率，改善生活质量。\
近年来，深度学习技术取得了突破性进展，为各行各业带来了新的机遇。"
    
    f10_h = extract_features_10d(human_text)
    f15_h = extract_features_15d(human_text)
    f10_a = extract_features_10d(ai_text)
    f15_a = extract_features_15d(ai_text)
    
    print("=" * 60)
    print("10D vs 15D 特征对比")
    print("=" * 60)
    print(f"\nHuman text (10D): {f10_h}")
    print(f"Human text (15D): {f15_h}")
    print(f"\nAI text (10D): {f10_a}")
    print(f"AI text (15D): {f15_a}")
    
    # 差异分析
    diff10 = abs(f10_h - f10_a)
    diff15 = abs(f15_h - f15_a)
    
    print(f"\n10D 差异 |Human-AI|: {diff10[:10]}")
    print(f"15D 新增差异 |Human-AI|: {diff15[10:]}")
    print(f"\n15D新增特征说明:")
    print(f"  10.句法嵌套深度: Human={f15_h[10]:.3f} AI={f15_a[10]:.3f}")
    print(f"  11.从句密度: Human={f15_h[11]:.3f} AI={f15_a[11]:.3f}")
    print(f"  12.被动句比例: Human={f15_h[12]:.3f} AI={f15_a[12]:.3f}")
    print(f"  13.修饰语密度: Human={f15_h[13]:.3f} AI={f15_a[13]:.3f}")
    print(f"  14.句法多样性: Human={f15_h[14]:.3f} AI={f15_a[14]:.3f}")
