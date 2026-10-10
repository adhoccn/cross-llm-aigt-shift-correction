# -*- coding: utf-8 -*-
"""
改写攻击损失的机制分解（E33 的补充诊断）。

待解释的两个现象：
  (1) char 特征上 para_ai 的未校正 F1 反而高于 orig：改写把 AI 文本推向人类
      侧，顺带抵消了原本的漂移，等于替检测器改了阈值。
  (2) para_all 下增益最小：两侧同时移动，类间间隔未必变大，但判别信息被破坏。

分解思路：类条件对齐对每个样本施加的位移是其后验概率的单调函数，故校正等价
于对分数做单调重标定 —— AUC 不变，收益全部来自决策阈值。于是把攻击损失拆成
两块：

      损失_total  = F1(orig, 未校正) − F1(cond, 未校正)
      可修部分    = F1(cond, 最优阈值) − F1(cond, 未校正)
      不可修部分  = F1(orig, 未校正) − F1(cond, 最优阈值)

最优阈值由真值标签选定，是任意单调重标定类方法的上界。若 EM 接近该上界，
说明残留损失来自排序 / 信息破坏，而非漂移。

Ξ_eff 的定义只度量 AI 类均值漂移，而 para_all 下人类侧也在移动，故同时报告
xi_hu_eff 与目标域实测类间间隔 Δ_T = uᵀ(μ_T^{AI} − μ_T^{HU}) 及信噪比
Δ_T/σ_T。

输出：results/E33b_paraphrase_diag.json
"""
from __future__ import annotations
import io, os, sys, json, time, importlib.util, argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core.invariance import estimate_Xi

import E33_paraphrase_attack as E33
import E17_crossllm_ghostbuster as E17

_spec = importlib.util.spec_from_file_location('e26', HERE / 'E26_tta.py')
E26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E26)

N_CG = E33.N_CG


def best_f1_by_threshold(y, s):
    """用真值标签扫阈值，返回 (最优 macro-F1, 最优阈值)"""
    grid = np.unique(np.quantile(s, np.linspace(0.001, 0.999, 999)))
    best, bt = -1.0, 0.5
    for t in grid:
        v = f1_score(y, (s > t).astype(int), average='macro')
        if v > best:
            best, bt = float(v), float(t)
    return best, bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per', type=int, default=200)
    ap.add_argument('--feats', default='15D,char_2_4')
    ap.add_argument('--targets', default='deepseek,glm')
    ap.add_argument('--conds', default='orig,para_ai,para_all')
    ap.add_argument('--out', default=str(HERE / 'results' / 'E33b_paraphrase_diag.json'))
    args = ap.parse_args()

    t0 = time.time()
    corpus = E33.load_corpus()
    hc3 = E17.build_hc3_index()
    cg_h, cg_a, _ = E17.build_pairs('chatgpt', hc3)
    tr_texts = list(cg_h[:N_CG]) + list(cg_a[:N_CG])
    ytr = np.array([0] * N_CG + [1] * N_CG)

    out = {'protocol': ('机制分解：AUC 在类条件对齐前后不变（位移是后验的单调函数），'
                        '故把攻击损失拆为「阈值/漂移可修部分」与「排序/信息不可修部分」。'
                        '最优阈值用目标批真值标签选取，作为任意单调重标定方法的上界。'),
           'n_per': args.n_per, 'conds': args.conds.split(','), 'feature_sets': {}}

    for fs_name, ctor in E26.FEATURE_SETS:
        if fs_name not in args.feats.split(','):
            continue
        f = ctor()
        # 与 E19/E26 的规范管线一致：中位数填补 + 在训练集上拟合的标准化
        A0 = np.asarray(f.fit(tr_texts).transform(tr_texts), dtype=np.float64)
        imp = SimpleImputer(strategy='median').fit(A0)
        sc = StandardScaler().fit(imp.transform(A0))
        A = sc.transform(imp.transform(A0))
        mu0, mu1 = A[ytr == 0].mean(axis=0), A[ytr == 1].mean(axis=0)
        clf = LogisticRegression(max_iter=2000).fit(A, ytr)
        # 源域判别方向 u 与沿该方向的类内标准差
        d = A.shape[1]
        SW = np.zeros((d, d))
        for c, Xc in [(0, A[ytr == 0]), (1, A[ytr == 1])]:
            Xc0 = Xc - Xc.mean(axis=0)
            SW += Xc0.T @ Xc0
        SW /= max(len(A) - 2, 1)
        SW += 1e-2 * (np.trace(SW) / max(d, 1)) * np.eye(d)
        SWi = np.linalg.pinv(SW)
        dmu = mu1 - mu0
        w = SWi @ dmu
        Delta = float(np.sqrt(max(dmu @ SWi @ dmu, 0.0)))
        u = w / Delta if Delta > 1e-9 else np.zeros_like(w)
        sig_S = float(np.sqrt(0.5 * (np.var((A[ytr == 0] - mu0) @ u, ddof=1)
                                     + np.var((A[ytr == 1] - mu1) @ u, ddof=1))))
        out['feature_sets'][fs_name] = {'Delta_src': round(Delta, 4),
                                        'sigma_src': round(sig_S, 4),
                                        'SNR_src': round(Delta / max(sig_S, 1e-9), 4),
                                        'targets': {}}

        for tgt in args.targets.split(','):
            out['feature_sets'][fs_name]['targets'][tgt] = {}
            for cond in args.conds.split(','):
                hs, ais = E33.build_condition(corpus, hc3, tgt, cond, args.n_per)
                te = list(hs) + list(ais)
                yte = np.array([0] * len(hs) + [1] * len(ais))
                B = sc.transform(imp.transform(np.asarray(f.transform(te), dtype=np.float64)))
                Bai, Bhu = B[yte == 1], B[yte == 0]
                xi = estimate_Xi(A, ytr, Bai, Bhu)
                muA, muH = Bai.mean(axis=0), Bhu.mean(axis=0)
                Delta_T = float(u @ (muA - muH))
                sig_T = float(np.sqrt(0.5 * (np.var((Bhu - muH) @ u, ddof=1)
                                             + np.var((Bai - muA) @ u, ddof=1))))
                s = clf.decision_function(B)
                pb = clf.predict_proba(B)[:, 1]
                f1_none = float(f1_score(yte, (pb > 0.5).astype(int), average='macro'))
                f1_opt, thr_opt = best_f1_by_threshold(yte, s)
                B_em = E26.tta_em(B, clf, mu0, mu1, 3)
                f1_em = float(f1_score(yte, clf.predict(B_em), average='macro'))
                cell = {
                    'F1_none': round(f1_none, 4),
                    'F1_optthr': round(f1_opt, 4), 'thr_opt': round(thr_opt, 4),
                    'F1_em': round(f1_em, 4),
                    'fixable': round(f1_opt - f1_none, 4),
                    'unfixable': round(max(0.0, f1_opt - f1_em), 4),
                    'em_vs_opt': round(f1_em - f1_opt, 4),
                    'AUC': round(float(roc_auc_score(yte, pb)), 4),
                    'xi_ai': round(float(xi['xi_eff']), 4),
                    'xi_hu': round(float(xi['xi_hu_eff']), 4),
                    'Delta_T': round(Delta_T, 4), 'sigma_T': round(sig_T, 4),
                    'SNR_T': round(Delta_T / max(sig_T, 1e-9), 4),
                    'pseudo_acc': round(float(((pb > 0.5).astype(int) == yte).mean()), 4),
                    'n_texts': int(len(yte)),
                }
                out['feature_sets'][fs_name]['targets'][tgt][cond] = cell

    out['runtime_sec'] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    io.open(args.out, 'w', encoding='utf-8').write(
        json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
