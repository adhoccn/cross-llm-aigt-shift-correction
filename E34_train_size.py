# -*- coding: utf-8 -*-
"""
E34 —— 训练规模敏感性
==================================================================
问题：主实验固定训练规模为 ChatGPT-only 800 组，需说明该选择不影响结论。

做法：训练规模 n_tr ∈ {200, 400, 800, 1200}（ChatGPT-only，人类/AI 各半），
特征管线与 E19/E26 一致（中位数填补 + 训练集上拟合的标准化），
在全部 5 个目标 LLM 的全量目标批（各 1000 组）上评估无校正 macro-F1，
5-fold×5seeds（fold 划分在目标批上，分类器固定）。
同时报告源域留出集（训练未使用的 ChatGPT 样本）的 in-domain F1。
"""
from __future__ import annotations
import io, os, sys, json, time
from pathlib import Path
import numpy as np

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import E17_crossllm_ghostbuster as E17
import importlib.util
_spec = importlib.util.spec_from_file_location('e26', HERE / 'E26_tta.py')
E26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E26)

SEEDS = [42, 123, 456, 789, 2024]
SIZES = [200, 400, 800, 1000]   # 配对数（每类 n_tr 条；800 即论文 §3.3 的固定规模）
TARGETS = ['deepseek', 'qwen', 'glm', 'gpt-4o', 'llama3']
OUT = HERE / 'results' / 'E34_train_size.json'


def log(*a):
    print('[%s]' % time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    t0 = time.time()
    hc3 = E17.build_hc3_index()
    cg_h, cg_a, _ = E17.build_pairs('chatgpt', hc3)
    log('ChatGPT 可用配对：%d' % len(cg_h))

    out = {'protocol': ('训练规模敏感性：ChatGPT-only 训练规模 {200,400,800,1200}（人类/AI 各半），'
                        '特征管线与 E19/E26 一致；目标批为各 LLM 全量 1000 组，'
                        '5-fold×5seeds（fold 划分在目标批上，分类器固定）。'),
           'sizes': SIZES, 'feature_sets': {}}

    for fs_name, ctor in E26.FEATURE_SETS:
        log('── %s ──' % fs_name)
        out['feature_sets'][fs_name] = {'targets': {t: {} for t in TARGETS},
                                        'in_domain': {}}
        for n_tr in SIZES:
            tr = list(cg_h[:n_tr]) + list(cg_a[:n_tr])
            ytr = np.array([0] * n_tr + [1] * n_tr)
            f = ctor()
            A0 = np.asarray(f.fit(tr).transform(tr), dtype=np.float64)
            imp = SimpleImputer(strategy='median').fit(A0)
            sc = StandardScaler().fit(imp.transform(A0))
            A = sc.transform(imp.transform(A0))
            clf = LogisticRegression(max_iter=2000).fit(A, ytr)
            # 各目标 LLM 全量目标批
            for tgt in TARGETS:
                hs, ais, _ = E17.build_pairs(tgt, hc3)
                te = list(hs) + list(ais)
                yte = np.array([0] * len(hs) + [1] * len(ais))
                B = sc.transform(imp.transform(np.asarray(f.transform(te), dtype=np.float64)))
                p = clf.predict(B)
                folds = []
                for seed in SEEDS:
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                    for _, idx in skf.split(np.zeros(len(yte)), yte):
                        folds.append(float(f1_score(yte[idx], p[idx], average='macro')))
                cell = {'F1': round(float(np.mean(folds)), 4),
                        'std': round(float(np.std(folds)), 4)}
                out['feature_sets'][fs_name]['targets'][tgt]['n_%d' % n_tr] = cell
                log('  %-8s n_tr=%-5d F1=%.4f±%.4f' % (tgt, n_tr, cell['F1'], cell['std']))
            # in-domain（留出：训练未用的 ChatGPT 样本，至多每类 400、不足则用尽）
            n_hold = min(400, len(cg_h) - n_tr)
            if n_hold < 50:
                out['feature_sets'][fs_name]['in_domain']['n_%d' % n_tr] = None
            else:
                hold = list(cg_h[n_tr:n_tr + n_hold]) + list(cg_a[n_tr:n_tr + n_hold])
                yho = np.array([0] * n_hold + [1] * n_hold)
                Bh = sc.transform(imp.transform(np.asarray(f.transform(hold), dtype=np.float64)))
                v = float(f1_score(yho, clf.predict(Bh), average='macro'))
                out['feature_sets'][fs_name]['in_domain']['n_%d' % n_tr] = round(v, 4)
                log('  in-domain n_tr=%-5d F1=%.4f' % (n_tr, v))

    out['runtime_sec'] = round(time.time() - t0, 1)
    io.open(OUT, 'w', encoding='utf-8').write(json.dumps(out, ensure_ascii=False, indent=1))
    log('写出 %s (%.0fs)' % (OUT, out['runtime_sec']))


if __name__ == '__main__':
    main()
