# -*- coding: utf-8 -*-
"""
数据加载与划分：HC3-Chinese + 多LLM扩展数据
划分协议对齐论文 §3.4.1（严格三方划分）与 §3.4.2（Pair-Stratified 5-fold）

关键：问题级别隔离（question-level isolation）——同一 question 衍生的
human / AI / 各LLM 文本必须落在同一折，否则会造成性能高估（论文 §4.1.2）。
"""
import json
import io
import os
import random
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def load_hc3(domain=None, path=None):
    """返回 list[dict]: {domain, qid, question, human, chatgpt}"""
    path = path or os.path.join(DATA_DIR, 'all.jsonl')
    recs = []
    with io.open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            src = d.get('source', 'unknown')
            if domain and src != domain:
                continue
            human = d.get('human_answers') or ['']
            ai = d.get('chatgpt_answers') or ['']
            recs.append({
                'domain': src,
                'qid': '%s#%d' % (src, i),
                'question': d.get('question', ''),
                'human': human[0] if human else '',
                'chatgpt': ai[0] if ai else '',
                'human_all': human,
                'chatgpt_all': ai,
            })
    return recs


def load_llm_jsonl(path, llm_name):
    """加载自建多LLM数据：每行 {question, answer, domain?}
    返回 list[dict]: {domain, qid, question, <llm_name>: answer}"""
    out = []
    with io.open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            out.append({
                'domain': d.get('domain', 'unknown'),
                'qid': '%s#%d' % (llm_name, i),
                'question': d.get('question', ''),
                llm_name: d.get('answer', ''),
            })
    return out


def make_binary_set(recs, domain=None, human_field='human', ai_field='chatgpt',
                    max_n=None, seed=42):
    """构造二分类样本：(text, label, qid, domain)   label 1=AI, 0=human"""
    rows = []
    for r in recs:
        if domain and r['domain'] != domain:
            continue
        h, a = r.get(human_field, ''), r.get(ai_field, '')
        if h:
            rows.append({'text': h, 'label': 0, 'qid': r['qid'], 'domain': r['domain']})
        if a:
            rows.append({'text': a, 'label': 1, 'qid': r['qid'], 'domain': r['domain']})
    if max_n and len(rows) > max_n:
        random.Random(seed).shuffle(rows)
        rows = rows[:max_n]
    return rows


def pair_stratified_kfold(rows, k=5, seed=42):
    """按 qid 分组做 K 折，保证同一 question 的样本不跨折；同时按标签分层。
    返回 list[(train_idx, test_idx)]"""
    by_q = defaultdict(list)
    for i, r in enumerate(rows):
        by_q[r['qid']].append(i)
    qids = sorted(by_q.keys())
    rng = random.Random(seed)

    # 先按每个 question 的标签构成分桶，再做分层分配
    def key(q):
        labels = tuple(sorted({rows[i]['label'] for i in by_q[q]}))
        return labels
    buckets = defaultdict(list)
    for q in qids:
        buckets[key(q)].append(q)
    folds = [[] for _ in range(k)]
    for _, qs in sorted(buckets.items()):
        rng.shuffle(qs)
        for j, q in enumerate(qs):
            folds[j % k].append(q)

    out = []
    for j in range(k):
        test_q = set(folds[j])
        test_idx = [i for q in test_q for i in by_q[q]]
        train_idx = [i for jj in range(k) if jj != j for q in folds[jj] for i in by_q[q]]
        out.append((train_idx, test_idx))
    return out


def group_kfold_by_question(rows, k=5, seed=42):
    """同 pair_stratified_kfold 的别名，语义更明确"""
    return pair_stratified_kfold(rows, k=k, seed=seed)


def domain_counts(recs):
    c = defaultdict(int)
    for r in recs:
        c[r['domain']] += 1
    return dict(sorted(c.items(), key=lambda x: -x[1]))
