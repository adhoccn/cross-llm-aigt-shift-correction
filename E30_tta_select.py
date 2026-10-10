# -*- coding: utf-8 -*-
"""
单因子 vs 双因子模型：为什么残留漂移更小却不一定精度更高。

残留漂移与精度并非单调对应：决定精度的还有对齐操作注入的类内方差。
把测试样本投影到分类器线性判别 z = βᵀx + b（边界 z=0）：
    每个类的正确率 ≈ Φ( mean_c(z) / sd_c(z) )            （高斯近似）
    → 由 (mean_0, sd_0, mean_1, sd_1) 解析预测 macro-F1
mean_c 受均值漂移影响，sd_c 受注入噪声影响，二者此消彼长。

实验：
  1. 用真值标签估计 (mean, sd)，检验双因子模型能否解释全部变体的 F1 排序
  2. 用伪标签（无监督）估计同一统计量，检验该准则能否在无标签下自动选
     (λ, keep)；若 auto 选出的配置接近 oracle 最优，则 TTA 无需标签调参
  3. 对照只看均值残留的单因子模型，比较 MAE
"""
from __future__ import annotations
import io, os, sys, json, time, argparse, importlib.util, itertools
from pathlib import Path
import numpy as np

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core.invariance import estimate_Xi
from scipy.stats import norm
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

_spec = importlib.util.spec_from_file_location('e26', HERE / 'E26_tta.py')
E26 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(E26)
_spec2 = importlib.util.spec_from_file_location('e17', HERE / 'E17_crossllm_ghostbuster.py')
E17 = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(E17)
_spec3 = importlib.util.spec_from_file_location('e28', HERE / 'E28_tta_lambda.py')
E28 = importlib.util.module_from_spec(_spec3); _spec3.loader.exec_module(E28)
_spec4 = importlib.util.spec_from_file_location('e29', HERE / 'E29_tta_robust.py')
E29 = importlib.util.module_from_spec(_spec4); _spec4.loader.exec_module(E29)

SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5
N_CG = 800
OUT_PATH = HERE / 'results' / 'E30_tta_select.json'


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def pred_f1(z, yhat, n1=None):
    """
    由 score 分布解析预测 macro-F1：
      每类正确率 = Φ(mean_c/sd_c)（边界 0，类 1 在正侧）
      → TP/FP/FN → precision/recall → macro-F1
    yhat: 用于估计 (mean, sd) 的分组标签（真值=机制验证；伪标签=无监督使用）
    """
    m1 = float(z[yhat == 1].mean()); s1 = float(z[yhat == 1].std())
    m0 = float(z[yhat == 0].mean()); s0 = float(z[yhat == 0].std())
    if s1 < 1e-9:
        s1 = 1e-9
    if s0 < 1e-9:
        s0 = 1e-9
    # 方向处理：允许 m1 < m0（严重错位）
    sgn = 1.0 if m1 >= m0 else -1.0
    e1 = float(norm.cdf(-sgn * (m1 - 0.0) / s1))     # 类1 被判为 0
    e0 = float(norm.cdf(sgn * (m0 - 0.0) / s0))      # 类0 被判为 1
    if abs(m1) < abs(m0):
        # 边界相对位置按两类均值中点更稳（类均值可能被 TTA 整体平移）
        pass
    n = 0.5
    tp = n * (1 - e1); fp = n * e0; fn = n * e1; tn = n * (1 - e0)
    p1 = tp / max(tp + fp, 1e-9); r1 = tp / max(tp + fn, 1e-9)
    f1_1 = 2 * p1 * r1 / max(p1 + r1, 1e-9)
    p0 = tn / max(tn + fn, 1e-9); r0 = tn / max(tn + fp, 1e-9)
    f1_0 = 2 * p0 * r0 / max(p0 + r0, 1e-9)
    return 0.5 * (f1_0 + f1_1), dict(m1=round(m1, 3), s1=round(s1, 3),
                                     m0=round(m0, 3), s0=round(s0, 3),
                                     e1=round(e1, 4), e0=round(e0, 4))


def pred_gmm(z):
    """无标签版本：在测试 score 上拟合一维两分量高斯混合 → 解析预测 macro-F1"""
    try:
        from sklearn.mixture import GaussianMixture
        zz = np.asarray(z, dtype=np.float64).reshape(-1, 1)
        gm = GaussianMixture(n_components=2, random_state=0, max_iter=300,
                             means_init=np.array([[zz.min()], [zz.max()]])).fit(zz)
        order = np.argsort(gm.means_.ravel())
        mu0, mu1 = gm.means_.ravel()[order]
        s0 = np.sqrt(max(gm.covariances_.ravel()[order[0]], 1e-9))
        s1 = np.sqrt(max(gm.covariances_.ravel()[order[1]], 1e-9))
        w0, w1 = gm.weights_.ravel()[order]
        # 均值大的一侧为 AI
        e1 = float(norm.cdf(-mu1 / s1))
        e0 = float(norm.cdf(mu0 / s0))
        n0, n1 = w0, w1
        tp = n1 * (1 - e1); fp = n0 * e0; fn = n1 * e1; tn = n0 * (1 - e0)
        p1 = tp / max(tp + fp, 1e-9); r1 = tp / max(tp + fn, 1e-9)
        f1_1 = 2 * p1 * r1 / max(p1 + r1, 1e-9)
        p0 = tn / max(tn + fn, 1e-9); r0 = tn / max(tn + fp, 1e-9)
        f1_0 = 2 * p0 * r0 / max(p0 + r0, 1e-9)
        return 0.5 * (f1_0 + f1_1), dict(mu1=round(float(mu1), 3), mu0=round(float(mu0), 3),
                                         s1=round(float(s1), 3), s0=round(float(s0), 3))
    except Exception:
        return float('nan'), {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='deepseek,glm')
    ap.add_argument('--mix', default='0')
    ap.add_argument('--feats', default='15D,char_2_4')
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--out', default=str(OUT_PATH))
    ap.add_argument('--crit', default='gmm', choices=['stale', 'self', 'gmm', 'gmm+nobal', 'entropy'])
    args = ap.parse_args()
    OUT = Path(args.out)
    targets = args.targets.split(',')
    alphas = [float(x) for x in args.mix.split(',')]
    keeps = [0.3, 0.5, 0.7]
    lams = [0.5, 1.0]

    t0 = time.time()
    hc3_idx = E17.build_hc3_index()
    data = {}
    for m in ['chatgpt', 'deepseek'] + targets:
        hs, ais, _ = E17.build_pairs(m, hc3_idx)
        data[m] = {'human': hs, 'ai': ais}
        log('  %-10s n=%d' % (m, len(ais)))

    cg, ds = data['chatgpt'], data['deepseek']
    rows = []
    out = {'feature_sets': {}, 'protocol': '双因子(均值+方差)模型 vs 单因子(仅均值)模型；auto=伪标签选参'}

    for fs_name in args.feats.split(','):
        if fs_name not in E28.FEATURES:
            continue
        log('── %s ──' % fs_name)
        fs_out = {}
        for alpha in alphas:
            n_ds = 0 if alpha <= 0 else int(round(N_CG * alpha / (1 - alpha)))
            tr_h = cg['human'][:N_CG] + ds['human'][:n_ds]
            tr_a = cg['ai'][:N_CG] + ds['ai'][:n_ds]
            tr_texts = list(tr_h) + list(tr_a)
            ytr = np.array([0] * len(tr_h) + [1] * len(tr_a))
            f, sc, A, clf, mu_tr, sd_tr, mu0_tr, mu1_tr = E29.build_pipe(fs_name, tr_texts, ytr)
            log('  train acc=%.4f dim=%d' % (clf.score(A, ytr), A.shape[1]))

            a_out = {'dim': int(A.shape[1]), 'n_ds': n_ds}
            for tgt in targets:
                th, ta = data[tgt]['human'], data[tgt]['ai']
                if tgt == 'deepseek' and n_ds > 0:
                    th, ta = th[n_ds:], ta[n_ds:]
                te_texts = list(th) + list(ta)
                yte = np.array([0] * len(th) + [1] * len(ta))
                B0 = np.asarray(f.transform(te_texts), dtype=np.float64) \
                    if not isinstance(f, E28.CachedRoberta) else f.embed('tgt_%s' % tgt, te_texts)
                B0 = sc.transform(np.nan_to_num(B0, nan=0.0, posinf=0.0, neginf=0.0))
                yhat0 = clf.predict(B0)

                var = {'none': B0, 'bn': E26.tta_bn(B0, mu_tr, sd_tr), 'bnc': E29._bnc(B0, mu_tr)}
                for lam in lams:
                    var['em_l%g' % lam] = E29.tta_robust(B0, clf, mu_tr, mu0_tr, mu1_tr,
                                                         args.iters, lam, 'em')
                    for k in keeps:
                        var['conf_l%g_k%.2f' % (lam, k)] = E29.tta_robust(
                            B0, clf, mu_tr, mu0_tr, mu1_tr, args.iters, lam, 'conf', k)

                cell = {}
                for v, B in var.items():
                    z = clf.decision_function(B)
                    p = clf.predict(B)
                    pb = clf.predict_proba(B)[:, 1]
                    pf = []
                    for seed in SEEDS:
                        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                        for _, idx in skf.split(np.zeros(len(yte)), yte):
                            pf.append(float(f1_score(yte[idx], p[idx], average='macro')))
                    fobs = float(np.mean(pf))
                    f_true, stat_true = pred_f1(z, yte)
                    f_auto, stat_auto = pred_f1(z, yhat0)
                    f_self, stat_self = pred_f1(z, p)                    # 准则③：对齐后自标签
                    f_gmm, stat_gmm = pred_gmm(z)                        # 准则④：一维两分量 GMM
                    xi = estimate_Xi(A, ytr, B[yte == 1], B[yte == 0])['xi_eff']
                    ent = float(-np.mean(pb * np.log(np.clip(pb, 1e-9, 1)) +
                                         (1 - pb) * np.log(np.clip(1 - pb, 1e-9, 1))))
                    bal = float(abs(pb.mean() - 0.5))
                    cell[v] = {'F1_obs': round(fobs, 4),
                               'F1_pred_true': round(f_true, 4),
                               'F1_pred_stale': round(f_auto, 4),
                               'F1_pred_self': round(f_self, 4),
                               'F1_pred_gmm': round(f_gmm, 4),
                               'entropy': round(ent, 4), 'imbalance': round(bal, 4),
                               'Xi_eff': round(float(xi), 4),
                               'stat_auto': stat_auto, 'per_fold': [round(x, 4) for x in pf]}
                    rows.append(dict(fs=fs_name, tgt=tgt, alpha=alpha, variant=v, obs=fobs,
                                     two_f=f_true, two_f_auto=f_auto, xi=float(xi),
                                     F1_pred_self=cell[v]['F1_pred_self'],
                                     F1_pred_gmm=cell[v]['F1_pred_gmm'],
                                     F1_pred_stale=cell[v]['F1_pred_stale'],
                                     entropy=cell[v]['entropy'], imbalance=cell[v]['imbalance']))
                a_out[tgt] = cell
                # auto 选参 vs oracle 选参
                crit = {'stale': lambda v: cell[v]['F1_pred_stale'],
                        'self': lambda v: cell[v]['F1_pred_self'],
                        'gmm': lambda v: cell[v]['F1_pred_gmm'],
                        'gmm+nobal': lambda v: (cell[v]['F1_pred_gmm']
                                                if cell[v]['imbalance'] < 0.08
                                                else -1.0),
                        'entropy': lambda v: -cell[v]['entropy']}
                best_auto = max((v for v in cell), key=crit.get(args.crit, crit['gmm']))
                best_orac = max((v for v in cell), key=lambda v: cell[v]['F1_obs'])
                log('   α=%.2f %-9s none=%.4f  auto→%-14s(%.4f)  oracle→%-14s(%.4f)  (%.0fs)'
                    % (alpha, tgt, cell['none']['F1_obs'], best_auto, cell[best_auto]['F1_obs'],
                       best_orac, cell[best_orac]['F1_obs'], time.time() - t0))
                a_out[tgt + '_choice'] = {'auto': best_auto, 'oracle': best_orac,
                                          'auto_F1': cell[best_auto]['F1_obs'],
                                          'oracle_F1': cell[best_orac]['F1_obs'],
                                          'none_F1': cell['none']['F1_obs']}
            fs_out['alpha_%g' % alpha] = a_out
        out['feature_sets'][fs_name] = fs_out

    # ── 汇总：三个模型的预测力 ──
    X_t = np.array([r['two_f'] for r in rows])
    X_a = np.array([r['two_f_auto'] for r in rows])
    X_x = np.array([r['xi'] for r in rows])
    Y = np.array([r['obs'] for r in rows])
    for nm in ('F1_pred_self', 'F1_pred_gmm', 'F1_pred_stale'):
        Xp = np.array([r[nm] for r in rows])
        rr, pp = sps.pearsonr(Xp, Y)
        print('%-22s Pearson r=%+.4f (p=%.2g)  R²=%.4f  MAE=%.4f'
              % (nm, rr, pp, rr ** 2, float(np.mean(np.abs(Xp - Y)))))
    ent = np.array([r['entropy'] for r in rows]); bal = np.array([r['imbalance'] for r in rows])
    print('%-22s Pearson r=%+.4f (p=%.2g)' % ('负熵准则', -sps.pearsonr(ent, Y)[0], sps.pearsonr(ent, Y)[1]))
    print('%-22s Pearson r=%+.4f (p=%.2g)' % ('类平衡准则(-imbal)', sps.pearsonr(-bal, Y)[0], sps.pearsonr(-bal, Y)[1]))
    print('')
    print('=' * 104)
    print('E30：双因子 vs 单因子（n=%d 变体×目标）' % len(rows))
    print('=' * 104)
    pred = {'n_variants': len(rows)}
    for key, name, Xp in [('double', '双因子(真值标签)', X_t),
                          ('double_auto', '双因子(伪标签/无监督)', X_a),
                          ('single', '单因子 Ξ_eff 残留', X_x)]:
        if key == 'single':
            r_, p_ = sps.pearsonr(X_x, Y)
            r2 = sps.linregress(X_x, Y).rvalue ** 2
            mae = float(np.mean(np.abs(X_x - Y)))
        else:
            r_, p_ = sps.pearsonr(Xp, Y)
            r2 = r_ ** 2
            mae = float(np.mean(np.abs(Xp - Y)))
        print('%-22s Pearson r=%+.4f (p=%.2g)  R²=%.4f  MAE=%.4f'
              % (name, r_, p_, r2, mae))
        pred[key] = {'pearson_r': round(float(r_), 4), 'p': float(p_),
                     'R2': round(float(r2), 4), 'MAE': round(mae, 4)}
    out['predictive'] = pred
    print('')
    print('--- 自动选参 vs 最优 oracle ---')
    for fs_name in out['feature_sets']:
        for ak, a in out['feature_sets'][fs_name].items():
            for tgt in targets:
                ch = a.get(tgt + '_choice')
                if ch:
                    print('%-10s %-8s %-9s none=%.4f → auto[%s]=%.4f | oracle[%s]=%.4f  损失=%.4f'
                          % (fs_name, ak, tgt, ch['none_F1'], ch['auto'], ch['auto_F1'],
                             ch['oracle'], ch['oracle_F1'], ch['oracle_F1'] - ch['auto_F1']))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print('\n[SAVED] %s (%.0fs)' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
