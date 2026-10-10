# -*- coding: utf-8 -*-
"""
鲁棒 E 步：伪标签质量对 TTA 的影响。

oracle-γ 与实际 γ 之间存在量级差距，说明对齐公式本身几乎无损，损失来自
E 步：错分样本被推向错误方向。

设伪标签错误率 e，单次类条件校正实际只消除 (1−2e) 的差分漂移，同时在类内
注入 ~e·‖a‖² 的错向方差，净收益
        G ≈ f(Ξ_eff·(1−λ(1−2e))) − g(λ²·e·‖a‖²)
最优 λ* 与最优选通（只用高置信样本）均由该权衡决定。

变体：
   bnc     λ=0 批中心对齐（理论残留 = 1/2）
   em      软后验 + 全样本施加 + 迭代（基线）
   trim    类条件均值只用高置信样本估计
   conf    同上，且位移只施加在高置信样本上
   oracle  E 步用真值（上界）

判据：trim/conf 是否显著优于 em 并接近 oracle。
"""
from __future__ import annotations
import io, os, sys, json, time, argparse, importlib.util
from pathlib import Path
import numpy as np

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core.invariance import estimate_Xi

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from scipy import stats

_spec = importlib.util.spec_from_file_location('e26', HERE / 'E26_tta.py')
E26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E26)
_spec2 = importlib.util.spec_from_file_location('e17', HERE / 'E17_crossllm_ghostbuster.py')
E17 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(E17)
_spec3 = importlib.util.spec_from_file_location('e27', HERE / 'E27_roberta_tta.py')
E27 = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(E27)
_spec4 = importlib.util.spec_from_file_location('e28', HERE / 'E28_tta_lambda.py')
E28 = importlib.util.module_from_spec(_spec4)
_spec4.loader.exec_module(E28)

SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5
N_CG = 800
PI = 0.5
OUT_PATH = HERE / 'results' / 'E29_tta_robust.json'


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def tta_robust(B, clf, mu_tr, mu0_tr, mu1_tr, iters=3, lam=1.0,
               mode='em', keep=0.5, oracle_g=None):
    """
    mode: em     —— 软后验、全样本
          trim   —— 均值只用 top-keep 置信样本估计，位移仍施加全样本
          conf   —— 均值只用 top-keep，位移也只施加于 top-keep
    """
    Bc = _bnc(B, mu_tr)
    dmu_tr = mu1_tr - mu0_tr
    for it in range(iters):
        g = oracle_g.astype(np.float64) if oracle_g is not None else clf.predict_proba(Bc)[:, 1]
        if mode in ('trim', 'conf'):
            conf_score = np.abs(g - PI) if oracle_g is None else np.ones(len(g))
            k = max(int(round(keep * len(g))), 20)
            idx = np.argsort(-conf_score)[:k]
            mask = np.zeros(len(g), dtype=bool)
            mask[idx] = True
        else:
            mask = np.ones(len(g), dtype=bool)

        gw = np.where(mask, g, 0.0)
        s1 = gw.sum()
        s0 = np.where(mask, 1.0 - g, 0.0).sum()
        if s1 < 1e-6 or s0 < 1e-6:
            break
        mu1 = (Bc * gw[:, None]).sum(axis=0) / s1
        mu0 = (Bc * np.where(mask, 1.0 - g, 0.0)[:, None]).sum(axis=0) / s0
        a = dmu_tr - (mu1 - mu0)
        step = lam * a * (g - PI)[:, None]
        if mode == 'conf':
            step = step * mask[:, None]
        step = step - step.mean(axis=0)          # 锁住批中心
        Bc = Bc + step
        if oracle_g is not None:
            break
    return Bc


def _bnc(B, mu_tr):
    return B + (mu_tr - B.mean(axis=0))


def build_pipe(fs_name, tr_texts, ytr):
    f = E28.FEATURES[fs_name]()
    if isinstance(f, E28.CachedRoberta):
        A = f.embed('train_shared', tr_texts)
    else:
        A = np.asarray(f.fit(tr_texts).transform(tr_texts), dtype=np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    sc = StandardScaler().fit(A)
    A = sc.transform(A)
    clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0).fit(A, ytr)
    return f, sc, A, clf, A.mean(axis=0), A.std(axis=0), A[ytr == 0].mean(axis=0), A[ytr == 1].mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='deepseek,glm')
    ap.add_argument('--mix', default='0')
    ap.add_argument('--feats', default='15D,char_2_4')
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--keep', default='0.3,0.5,0.7')
    ap.add_argument('--out', default=str(OUT_PATH))
    args = ap.parse_args()
    OUT = Path(args.out)
    targets = args.targets.split(',')
    alphas = [float(x) for x in args.mix.split(',')]
    keeps = [float(x) for x in args.keep.split(',')]

    t0 = time.time()
    hc3_idx = E17.build_hc3_index()
    data = {}
    for m in ['chatgpt', 'deepseek'] + targets:
        hs, ais, _ = E17.build_pairs(m, hc3_idx)
        data[m] = {'human': hs, 'ai': ais}
        log('  %-10s n=%d' % (m, len(ais)))

    cg, ds = data['chatgpt'], data['deepseek']
    out = {'protocol': ('训练 ChatGPT-only(+α); 测试目标 LLM 整批无标签; '
                        'TTA: bnc / em / trim(k) / conf(k) / oracle; 5-fold×5seeds; macro-F1'),
           'iters': args.iters, 'keeps': keeps, 'feature_sets': {}}

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
            f, sc, A, clf, mu_tr, sd_tr, mu0_tr, mu1_tr = build_pipe(fs_name, tr_texts, ytr)
            log('  train acc=%.4f dim=%d' % (clf.score(A, ytr), A.shape[1]))

            a_out = {'n_ds': n_ds, 'dim': int(A.shape[1])}
            for tgt in targets:
                th, ta = data[tgt]['human'], data[tgt]['ai']
                if tgt == 'deepseek' and n_ds > 0:
                    th, ta = th[n_ds:], ta[n_ds:]
                te_texts = list(th) + list(ta)
                yte = np.array([0] * len(th) + [1] * len(ta))
                if isinstance(f, E28.CachedRoberta):
                    B0 = f.embed('tgt_%s' % tgt, te_texts)
                else:
                    B0 = np.asarray(f.transform(te_texts), dtype=np.float64)
                B0 = sc.transform(np.nan_to_num(B0, nan=0.0, posinf=0.0, neginf=0.0))

                variants = {
                    'none': B0,
                    'bn': E26.tta_bn(B0, mu_tr, sd_tr),
                    'bnc': _bnc(B0, mu_tr),
                    'em': tta_robust(B0, clf, mu_tr, mu0_tr, mu1_tr, args.iters, 1.0, 'em'),
                }
                for k in keeps:
                    variants['trim_%.2f' % k] = tta_robust(B0, clf, mu_tr, mu0_tr, mu1_tr,
                                                           args.iters, 1.0, 'trim', k)
                    variants['conf_%.2f' % k] = tta_robust(B0, clf, mu_tr, mu0_tr, mu1_tr,
                                                           args.iters, 1.0, 'conf', k)
                variants['oracle'] = tta_robust(B0, clf, mu_tr, mu0_tr, mu1_tr, 1, 1.0, 'em',
                                               oracle_g=yte)
                cell = {}
                for v, B in variants.items():
                    xi = estimate_Xi(A, ytr, B[yte == 1], B[yte == 0])['xi_eff']
                    p = clf.predict(B)
                    pb = clf.predict_proba(B)[:, 1]
                    pf = []
                    for seed in SEEDS:
                        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                        for _, idx in skf.split(np.zeros(len(yte)), yte):
                            pf.append(float(f1_score(yte[idx], p[idx], average='macro')))
                    cell[v] = {'F1_macro': round(float(np.mean(pf)), 4),
                               'std': round(float(np.std(pf)), 4),
                               'AUC': round(float(roc_auc_score(yte, pb)), 4),
                               'Xi_eff': round(float(xi), 4),
                               'per_fold_F1': [round(x, 4) for x in pf]}
                base = np.array(cell['none']['per_fold_F1'])
                for v in cell:
                    if v == 'none' or not isinstance(cell[v], dict):
                        continue
                    cur = np.array(cell[v]['per_fold_F1'])
                    cell[v]['delta'] = round(float(np.mean(cur - base)), 4)
                    cell[v]['p_paired_t'] = float(stats.ttest_rel(cur, base)[1])
                x0 = cell['none']['Xi_eff']
                for v in cell:
                    if isinstance(cell[v], dict):
                        cell[v]['resid'] = round(cell[v]['Xi_eff'] / x0, 4) if abs(x0) > 1e-6 else None
                cell['pseudo_err'] = round(1.0 - float((clf.predict(B0) == yte).mean()), 4)
                a_out[tgt] = cell
                best = max(((v, cell[v]['F1_macro']) for v in cell
                        if isinstance(cell[v], dict) and v != 'oracle'), key=lambda x: x[1])
                log('   α=%.2f %-9s none=%.4f bn=%.4f em=%.4f best=%s(%.4f) oracle=%.4f (%.0fs)'
                    % (alpha, tgt, cell['none']['F1_macro'], cell['bn']['F1_macro'],
                       cell['em']['F1_macro'], best[0], best[1], cell['oracle']['F1_macro'],
                       time.time() - t0))
            fs_out['alpha_%g' % alpha] = a_out
        out['feature_sets'][fs_name] = fs_out

    out['runtime_sec'] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    keys = ['none', 'bn', 'bnc', 'em'] + ['trim_%.2f' % k for k in keeps] + \
           ['conf_%.2f' % k for k in keeps] + ['oracle']
    print('')
    print('=' * 120)
    print('E29：鲁棒 E 步   α=%s  iters=%d' % (args.mix, args.iters))
    print('=' * 120)
    for fs_name in args.feats.split(','):
        fs = out['feature_sets'].get(fs_name, {})
        for alpha in alphas:
            a = fs.get('alpha_%g' % alpha)
            if not a:
                continue
            print('--- %s (dim=%s) | α=%.2f ---' % (fs_name, a.get('dim'), alpha))
            print('%-9s %s' % ('F1', ''.join('%11s' % k for k in keys)))
            for tgt in targets:
                print('%-9s %s' % (tgt, ''.join('%11.4f' % a[tgt][k]['F1_macro'] for k in keys)))
            print('%-9s %s' % ('Ξ残留比', ''.join(
                '%11.3f' % (a[targets[0]][k]['resid'] or 0) for k in keys)))
            print('')


if __name__ == '__main__':
    main()
