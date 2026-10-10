# -*- coding: utf-8 -*-
"""
估计环境散度 S_E 与类间散度 S_B 的重合程度（表 21 与图 10 的数据源）。

用法：
    python tools/se_sb_angle.py                默认 α=0.2，输出 results/E21_se_sb_angle.json
    python tools/se_sb_angle.py --alpha 0.2

在 α>0 的混合训练集上按 env = LLM×领域 估计三个散度矩阵，报告：
    tr(S_B)、tr(S_W)、tr(S_E)、tr(S_E)/tr(S_W)、tr(S_E)/tr(S_B)
    cos_top1  S_B 与 S_E 第一主方向的夹角余弦（绝对值）
    cos_top3  S_B 与 S_E 前三个主方向张成子空间的最大主余弦
前者衡量"主方向是否同向"，后者衡量"主子空间是否重合"。
"""
from __future__ import annotations
import io, os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import importlib.util
import E17_crossllm_ghostbuster as E17
from core.invariance import scatter_matrices
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

spec = importlib.util.spec_from_file_location(
    'e20', os.path.join(ROOT, 'E20_clida_main.py'))
E20 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E20)

N_CG = 800


def leading(M, k=1):
    _, V = np.linalg.eigh(M)
    return V[:, -k:]


def build(alpha):
    hc3 = E17.build_hc3_index()
    dom_idx = E20.build_domain_index()
    data = {}
    for m in ['chatgpt', 'deepseek']:
        hs, ais, dom = E20.build_pairs_with_domain(m, hc3, dom_idx)
        data[m] = {'human': hs, 'ai': ais, 'dom': dom}
    n_ds = 0 if alpha <= 0 else int(round(N_CG * alpha / (1 - alpha)))
    cg, ds = data['chatgpt'], data['deepseek']
    tr_h = cg['human'][:N_CG] + ds['human'][:n_ds]
    tr_a = cg['ai'][:N_CG] + ds['ai'][:n_ds]
    tr_d = cg['dom'][:N_CG] + ds['dom'][:n_ds]
    y = np.array([0] * len(tr_h) + [1] * len(tr_a))
    dom_full = list(tr_d) + list(tr_d)
    llm = [0] * N_CG + [1] * n_ds
    llm_full = list(llm) + list(llm)
    env = np.array(['%s|%s' % (a, b) for a, b in zip(llm_full, dom_full)])
    return list(tr_h) + list(tr_a), y, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.2)
    ap.add_argument('--feats', default='15D,char_2_4')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results',
                                                  'E21_se_sb_angle.json'))
    args = ap.parse_args()

    texts, y, env = build(args.alpha)
    out = {'alpha': args.alpha, 'n_cg': N_CG, 'env': 'LLM×domain',
           'feature_sets': {}}
    for name, ctor in E20.FEATURE_SETS:
        if name not in args.feats.split(','):
            continue
        A = np.asarray(ctor().fit(texts).transform(texts), dtype=np.float64)
        imp = SimpleImputer(strategy='median').fit(A)
        sc = StandardScaler().fit(imp.transform(A))
        A = sc.transform(imp.transform(A))
        S_B, S_W, S_E = scatter_matrices(A, y, env)
        tB, tW, tE = (float(np.trace(M)) for M in (S_B, S_W, S_E))
        vB, vE = leading(S_B), leading(S_E)
        cos1 = abs(float(vB[:, 0] @ vE[:, 0]))
        sing = np.linalg.svd(leading(S_B, 3).T @ leading(S_E, 3),
                             compute_uv=False)
        out['feature_sets'][name] = {
            'tr_S_B': round(tB, 6), 'tr_S_W': round(tW, 6), 'tr_S_E': round(tE, 6),
            'ratio_SE_SW': round(tE / max(tW, 1e-12), 6),
            'cos_top1': round(cos1, 6), 'cos_top3': round(float(sing.max()), 6),
            'ratio_SE_SB': round(tE / max(tB, 1e-12), 6)}
        print('%-10s tr_S_B=%.4f tr_S_W=%.4f tr_S_E=%.4f  cos1=%.3f cos3=%.3f'
              % (name, tB, tW, tE, cos1, float(sing.max())))
    with io.open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('wrote %s' % args.out)


if __name__ == '__main__':
    main()
