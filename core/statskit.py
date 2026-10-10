# -*- coding: utf-8 -*-
"""
统计推断工具：折级配对检验、非参数检验、样本级检验与多重比较校正。

折级检验与样本级检验的结论可能不一致：折级检验的 n 为折数（通常为 5），
Wilcoxon 在该样本量下 p 值存在下界 2/2^n；样本级 McNemar 与 bootstrap
不受此限制，用于规避折间不独立带来的伪重复。
"""
import math
import numpy as np
from scipy import stats


def paired_t(a, b):
    """折级配对 t 检验。返回 dict(t, p, mean_diff, ci95, d_z, n)"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = b - a
    n = len(d)
    md = d.mean()
    sd = d.std(ddof=1)
    se = sd / math.sqrt(n)
    t = md / se if se > 0 else float('inf')
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1)) if np.isfinite(t) else 0.0
    tcrit = stats.t.ppf(0.975, df=n - 1)
    ci = (md - tcrit * se, md + tcrit * se)
    d_z = t / math.sqrt(n) if np.isfinite(t) else float('inf')   # 关键换算
    return {'t': float(t), 'p': float(p), 'mean_diff': float(md),
            'ci95': (float(ci[0]), float(ci[1])), 'd_z': float(d_z), 'n': n}


def wilcoxon_paired(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    d = b - a
    if np.allclose(d, 0):
        return {'W': 0.0, 'p': 1.0, 'note': 'all differences are zero'}
    try:
        W, p = stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided')
    except Exception:
        return {'W': float('nan'), 'p': float('nan'), 'note': 'wilcoxon failed'}
    min_p = 2.0 / (2 ** len(d))          # n 折时最小非零 p（双侧 2/2^n）
    return {'W': float(W), 'p': float(p), 'min_attainable_p': min_p,
            'note': ('p 已到下界，无法达到 0.05' if p <= min_p + 1e-12 else '')}


def mcnemar(y_true, pred_a, pred_b):
    """样本级配对比较（解决折不独立的伪重复问题）
    pred_a / pred_b 为两套模型的 0/1 预测"""
    y = np.asarray(y_true); a = np.asarray(pred_a); b = np.asarray(pred_b)
    ok_a, ok_b = (a == y), (b == y)
    n01 = int(np.sum(~ok_a & ok_b))    # A 错 B 对
    n10 = int(np.sum(ok_a & ~ok_b))    # A 对 B 错
    n = n01 + n10
    if n == 0:
        return {'n01': n01, 'n10': n10, 'p': 1.0, 'method': 'mcnemar'}
    p = stats.binomtest(min(n01, n10), n, 0.5, alternative='two-sided').pvalue
    return {'n01': n01, 'n10': n10, 'n_discordant': n, 'p': float(p), 'method': 'mcnemar-exact'}


def bootstrap_diff(y_true, pred_a, pred_b, metric_fn, n_boot=2000, seed=42):
    """样本级 bootstrap：对 (pred_b - pred_a) 的指标差做重采样 CI"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true); a = np.asarray(pred_a); b = np.asarray(pred_b)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(metric_fn(y[idx], b[idx]) - metric_fn(y[idx], a[idx]))
    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        return {'mean_diff': float('nan'), 'ci95': (float('nan'), float('nan'))}
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {'mean_diff': float(diffs.mean()), 'ci95': (float(lo), float(hi)),
            'n_boot': len(diffs), 'p_gt0': float((diffs > 0).mean())}


def holm_bonferroni(pvals, alpha=0.05):
    """Holm-Bonferroni 逐步校正。返回 (reject_flags, adjusted_p)"""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    reject = adj <= alpha
    return reject.tolist(), adj.tolist()


def selfcheck_dz(t, d_z, n):
    """自洽性校验：d_z 必须等于 t/sqrt(n)"""
    expect = t / math.sqrt(n)
    return abs(expect - d_z) < 1e-6, expect
