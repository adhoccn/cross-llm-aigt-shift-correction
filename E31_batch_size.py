# -*- coding: utf-8 -*-
"""
批次规模敏感性：方法需要多大的一批无标签文本。

在每个 fold 内对目标批做 Bootstrap 重采样到规模 n，独立执行校正，考察增益
随 n 的变化。

输出：results/E31_batch_size.json
"""
import io, os, sys, json, time, argparse, importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from core.invariance import estimate_Xi
import E17_crossllm_ghostbuster as E17

_spec = importlib.util.spec_from_file_location('e26', HERE / 'E26_tta.py')
E26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E26)

N_CG = 800
N_FOLDS = 5
SEEDS = [42, 123, 456, 789, 2024]
PI = 0.5
OUT_PATH = HERE / 'results' / 'E31_batch_size.json'


def log(*a):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), ' '.join(str(x) for x in a)))


def _bnc(B, mu_tr):
    return B + (mu_tr - B.mean(axis=0))


def tta_em(B, clf, mu0_tr, mu1_tr, iters=1):
    """类条件对齐：先把批次中心对齐到训练混合中心，再做单步差分位移"""
    centre_tr = 0.5 * (mu0_tr + mu1_tr)
    Bc = B + (centre_tr - B.mean(axis=0))
    dmu_tr = mu1_tr - mu0_tr
    g = clf.predict_proba(Bc)[:, 1]
    w1 = g.sum()
    mu1 = (Bc * g[:, None]).sum(axis=0) / max(w1, 1e-9)
    w0 = (1.0 - g).sum()
    mu0 = (Bc * (1.0 - g)[:, None]).sum(axis=0) / max(w0, 1e-9)
    a = dmu_tr - (mu1 - mu0)
    return Bc + a * (g - PI)[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='deepseek,glm')
    ap.add_argument('--feats', default='15D,char_2_4')
    ap.add_argument('--sizes', default='50,100,200,500,1000,2000')
    ap.add_argument('--out', default=str(OUT_PATH))
    args = ap.parse_args()
    OUT = Path(args.out)
    targets = args.targets.split(',')
    sizes = [int(x) for x in args.sizes.split(',')]
    want = args.feats.split(',')

    t0 = time.time()
    log('批次规模敏感性 E31  sizes=%s' % sizes)
    hc3 = E17.build_hc3_index()
    data = {}
    for m in ['chatgpt'] + targets:
        hs, ais, _ = E17.build_pairs(m, hc3)
        data[m] = {'human': hs, 'ai': ais}
        log('  %-10s n=%d' % (m, len(ais)))

    fs_defs = dict(E26.FEATURE_SETS)
    out = {'protocol': 'ChatGPT-only(%d) 训练；目标批 Bootstrap 重采样到 n 后独立执行 UBC；'
                       '5-fold×5 seeds；macro-F1' % N_CG,
           'sizes': sizes, 'targets': targets, 'feature_sets': {}}

    cg = data['chatgpt']
    tr_texts_full = list(cg['human'][:N_CG]) + list(cg['ai'][:N_CG])
    ytr = np.array([0] * N_CG + [1] * N_CG)

    for fs_name in want:
        if fs_name not in fs_defs:
            continue
        log('特征集 %s' % fs_name)
        f = fs_defs[fs_name]()
        A = np.asarray(f.fit(tr_texts_full).transform(tr_texts_full), dtype=np.float64)
        mu1_tr = A[ytr == 1].mean(axis=0)
        mu0_tr = A[ytr == 0].mean(axis=0)
        out['feature_sets'][fs_name] = {}

        for tgt in targets:
            te = data[tgt]
            b_texts = np.array(list(te['human']) + list(te['ai']), dtype=object)
            bte = np.array([0] * len(te['human']) + [1] * len(te['ai']))
            B0 = np.asarray(f.transform(list(b_texts)), dtype=np.float64)
            out['feature_sets'][fs_name][tgt] = {}
            rng = np.random.default_rng(0)
            for n in sizes:
                pf_none, pf_em = [], []
                for seed in SEEDS:
                    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                    for tri, tei in skf.split(np.zeros(len(bte)), bte):
                        clf = LogisticRegression(max_iter=2000)
                        clf.fit(A, ytr)
                        Bsub = B0[tei]
                        ysub = bte[tei]
                        # Bootstrap 到规模 n
                        idx = rng.choice(len(Bsub), size=min(n, len(Bsub)),
                                         replace=(n > len(Bsub)))
                        Bs = Bsub[idx]
                        ys = ysub[idx]
                        pn = clf.predict(Bs)
                        pf_none.append(float(f1_score(ys, pn, average='macro')))
                        try:
                            Bc = tta_em(Bs, clf, mu0_tr, mu1_tr, iters=1)
                            pe = clf.predict(Bc)
                            pf_em.append(float(f1_score(ys, pe, average='macro')))
                        except Exception:
                            pf_em.append(float('nan'))
                fn = float(np.nanmean(pf_none))
                fe = float(np.nanmean(pf_em))
                fn_std = float(np.nanstd(pf_none))
                fe_std = float(np.nanstd(pf_em))
                # 逐次 bootstrap/fold 上校正反而变差的比率：小批次失稳的直接证据
                runs = [(a_, b_) for a_, b_ in zip(pf_none, pf_em) if not np.isnan(b_)]
                worse = sum(1 for a_, b_ in runs if b_ < a_ - 1e-9)
                out['feature_sets'][fs_name][tgt]['n_%d' % n] = {
                    'F1_none': round(fn, 4), 'F1_em': round(fe, 4),
                    'gain': round(fe - fn, 4),
                    'std_none': round(fn_std, 4), 'std_em': round(fe_std, 4),
                    'n_runs': len(pf_none),
                    'worse_rate': round(worse / max(1, len(runs)), 4)}
                log('   %-9s n=%-5d  none=%.4f±%.4f  em=%.4f±%.4f  增益=%+.4f  劣化率=%.0f%%'
                    % (tgt, n, fn, fn_std, fe, fe_std, fe - fn,
                       100.0 * worse / max(1, len(runs))))

    out['runtime_sec'] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    io.open(OUT, 'w', encoding='utf-8').write(json.dumps(out, ensure_ascii=False, indent=1))
    log('写出 %s  (%.1fs)' % (OUT, out['runtime_sec']))


if __name__ == '__main__':
    main()
