# -*- coding: utf-8 -*-
"""
跨 LLM 衰减率所需的同协议 in-domain 参照。

E16 的 in-domain（HC3 六域混合，1087 对）与 E17 的跨 LLM（collected 问题池，
1000 对）测试集的问题分布不同，直接相减得到的衰减率不严格。

方案：在 Mixed-20% 配置下训练（ChatGPT 前 800 对 + DeepSeek 前 200 对），
测试用 ChatGPT 的 held-out 后 200 对 —— 与跨 LLM 测试同训练、同协议、同
问题池，仅测试规模为 200 对。由此得到严格可比的 in-domain 参照。

输出：results/E18_indomain_ref.json
"""
from __future__ import annotations
import io, os, sys, json, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import importlib.util
_orig_stdout = sys.stdout
spec = importlib.util.spec_from_file_location('e17', HERE / 'E17_crossllm_ghostbuster.py')
E17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E17)
sys.stdout = _orig_stdout  # E17 顶层会替换 stdout，此处恢复

SEEDS = E17.SEEDS
METHODS = E17.METHODS


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def main():
    t0 = time.time()
    hc3_idx = E17.build_hc3_index()
    data = {}
    for m in ['chatgpt', 'deepseek']:
        hs, ais, qs = E17.build_pairs(m, hc3_idx)
        data[m] = {'human': hs, 'ai': ais}
        log('%s 匹配 %d 对' % (m, len(ais)))

    cg_h, cg_a = data['chatgpt']['human'], data['chatgpt']['ai']
    ds_h, ds_a = data['deepseek']['human'], data['deepseek']['ai']
    n_cg = len(cg_a)
    keep, mix_n = int(n_cg * 0.8), int(n_cg * 0.2)
    # 训练：ChatGPT 前 80% + DeepSeek 前 20%（与 E17 的 mixed20 完全一致）
    tr_texts = cg_h[:keep] + ds_h[:mix_n] + cg_a[:keep] + ds_a[:mix_n]
    ytr = np.array([0] * (keep + mix_n) + [1] * (keep + mix_n))
    # 测试：ChatGPT held-out 后 200 对
    te_texts = cg_h[keep:] + cg_a[keep:]
    yte = np.array([0] * len(cg_h[keep:]) + [1] * len(cg_a[keep:]))
    log('训练 %d 对（ChatGPT %d + DeepSeek %d），in-domain 测试 %d 对' %
        (len(ytr) // 2, keep, mix_n, len(yte) // 2))

    out = {'protocol': 'Mixed-20% 训练；测试 = ChatGPT held-out 后 200 对（与 E17 跨 LLM 同训练/同协议/同问题池）；'
                       'StratifiedKFold(5) x seeds{42,123,456,789,2024} = 25 evals；macro F1',
           'n_train_pairs': len(ytr) // 2, 'n_test_pairs': len(yte) // 2, 'methods': {}}

    for mname, vcls, ckind in METHODS:
        recs = []
        for seed in SEEDS:
            if mname == '15d_xgb':
                p, pb = E17.train_eval_15d(tr_texts, ytr, te_texts, yte, seed)
            else:
                p, pb = E17.train_eval_tfidf(vcls, ckind, tr_texts, ytr, te_texts, yte, seed)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for _, idx in skf.split(np.zeros(len(yte)), yte):
                recs.append(E17.metrics(yte[idx], p[idx], pb[idx]))
        out['methods'][mname] = {
            'mean': {k: round(float(np.mean([r[k] for r in recs])), 4) for k in ('F1_macro', 'F1', 'P', 'R', 'AUC')},
            'std': {k: round(float(np.std([r[k] for r in recs])), 4) for k in ('F1_macro', 'F1', 'P', 'R', 'AUC')},
        }
        log('%-18s macroF1=%.4f±%.4f  binF1=%.4f  (%.0fs)' %
            (mname, out['methods'][mname]['mean']['F1_macro'], out['methods'][mname]['std']['F1_macro'],
             out['methods'][mname]['mean']['F1'], time.time() - t0))

    out['runtime_sec'] = round(time.time() - t0, 1)
    op = HERE / 'results' / 'E18_indomain_ref.json'
    op.parent.mkdir(parents=True, exist_ok=True)
    with op.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('[SAVED] %s' % op)


if __name__ == '__main__':
    main()
