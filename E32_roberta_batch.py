# -*- coding: utf-8 -*-
"""
批次规模敏感性（冻结 RoBERTa 表征）。

回答 768 维深度表征上需要多大的一批无标签文本。与 E31 同构：同一 Bootstrap
重采样协议、同一单步 EM 校正、同一 5-fold×5seeds，仅把特征集换成冻结
RoBERTa，以便三条曲线直接并排比较。

RoBERTa 表征先过 StandardScaler（在训练集拟合），因为原始 768 维 embedding
量纲不利于 LR；15D / char 分支不做该处理，两者各自内部可比。

输出：results/E32_roberta_batch.json
"""
from __future__ import annotations
import io, os, sys, json, time, argparse, importlib.util
from pathlib import Path
import numpy as np

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import E17_crossllm_ghostbuster as E17

_spec = importlib.util.spec_from_file_location('e27', HERE / 'E27_roberta_tta.py')
E27 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E27)

N_CG = 800
N_FOLDS = 5
SEEDS = [42, 123, 456, 789, 2024]
PI = 0.5
CACHE = HERE / 'results' / '_e32_emb_cache.npz'
OUT_PATH = HERE / 'results' / 'E32_roberta_batch.json'


def log(*a):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), ' '.join(str(x) for x in a)), flush=True)


def tta_em(B, clf, mu0_tr, mu1_tr):
    """类条件对齐（与 E31 同一实现：单步、先对齐批中心、再做差分位移）"""
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
    ap.add_argument('--sizes', default='50,100,200,500,1000,2000')
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--out', default=str(OUT_PATH))
    args = ap.parse_args()
    OUT = Path(args.out)
    targets = args.targets.split(',')
    sizes = [int(x) for x in args.sizes.split(',')]

    t0 = time.time()
    log('E32 RoBERTa 批次规模敏感性  sizes=%s  targets=%s' % (sizes, targets))
    hc3 = E17.build_hc3_index()
    data = {}
    for m in ['chatgpt'] + targets:
        hs, ais, _ = E17.build_pairs(m, hc3)
        data[m] = {'human': hs, 'ai': ais}
        log('  %-10s n=%d' % (m, len(ais)))

    cg = data['chatgpt']
    tr_texts = list(cg['human'][:N_CG]) + list(cg['ai'][:N_CG])
    ytr = np.array([0] * N_CG + [1] * N_CG)

    # ---- embedding（缓存，避免重跑）----
    key_tr = 'train'
    need = [key_tr] + ['%s' % t for t in targets]
    cache = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        cache = {k: z[k] for k in z.files}
        log('  载入 embedding 缓存：%s' % list(cache.keys()))
    if not all(k in cache for k in need):
        feat = E27.RobertaFeat(max_len=args.max_len, batch=args.batch)
        feat.fit(tr_texts)
        if key_tr not in cache:
            cache[key_tr] = feat.transform(tr_texts)
            log('  train embed %s (%.0fs)' % (cache[key_tr].shape, time.time() - t0))
        for t in targets:
            if t not in cache:
                te = data[t]
                cache[t] = feat.transform(list(te['human']) + list(te['ai']))
                log('  %s embed %s (%.0fs)' % (t, cache[t].shape, time.time() - t0))
        np.savez_compressed(CACHE, **cache)
        log('  embedding 已缓存 -> %s' % CACHE.name)

    A_raw = np.asarray(cache[key_tr], dtype=np.float64)
    sc = StandardScaler().fit(A_raw)
    A = sc.transform(A_raw)
    mu1_tr = A[ytr == 1].mean(axis=0)
    mu0_tr = A[ytr == 0].mean(axis=0)

    clf = LogisticRegression(max_iter=3000, random_state=42, C=1.0).fit(A, ytr)
    log('  分类器就绪  train_acc=%.4f (%.0fs)' % (clf.score(A, ytr), time.time() - t0))

    out = {'protocol': ('冻结 chinese-roberta-wwm-ext mean-pooling(768d) + StandardScaler + LR；'
                        'ChatGPT-only(%d) 训练；目标批 Bootstrap 重采样到 n 后独立执行单步 EM 校正；'
                        '5-fold×5seeds；macro-F1' % N_CG),
           'model': 'hfl/chinese-roberta-wwm-ext (frozen)', 'max_len': args.max_len,
           'sizes': sizes, 'targets': targets, 'feature_sets': {'roberta_frozen': {}}}

    for tgt in targets:
        te = data[tgt]
        bte = np.array([0] * len(te['human']) + [1] * len(te['ai']))
        B0 = sc.transform(np.asarray(cache[tgt], dtype=np.float64))
        out['feature_sets']['roberta_frozen'][tgt] = {}
        rng = np.random.default_rng(0)
        for n in sizes:
            pf_none, pf_em = [], []
            for seed in SEEDS:
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                for _, tei in skf.split(np.zeros(len(bte)), bte):
                    Bsub = B0[tei]
                    ysub = bte[tei]
                    idx = rng.choice(len(Bsub), size=min(n, len(Bsub)),
                                     replace=(n > len(Bsub)))
                    Bs = Bsub[idx]
                    ys = ysub[idx]
                    pf_none.append(float(f1_score(ys, clf.predict(Bs), average='macro')))
                    Bc = tta_em(Bs, clf, mu0_tr, mu1_tr)
                    pf_em.append(float(f1_score(ys, clf.predict(Bc), average='macro')))
            fn = float(np.mean(pf_none)); fe = float(np.mean(pf_em))
            out['feature_sets']['roberta_frozen'][tgt]['n_%d' % n] = {
                'F1_none': round(fn, 4), 'F1_em': round(fe, 4), 'gain': round(fe - fn, 4)}
            log('   %-9s n=%-5d  none=%.4f  em=%.4f  增益=%+.4f  (%.0fs)'
                % (tgt, n, fn, fe, fe - fn, time.time() - t0))

    out['runtime_sec'] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    io.open(OUT, 'w', encoding='utf-8').write(json.dumps(out, ensure_ascii=False, indent=1))
    log('写出 %s  (%.1fs)' % (OUT, out['runtime_sec']))


if __name__ == '__main__':
    main()
