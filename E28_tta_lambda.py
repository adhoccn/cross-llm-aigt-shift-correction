# -*- coding: utf-8 -*-
"""
TTA 统一族：从批对齐到类条件对齐的连续插值，及其偏差-方差权衡。

动机：低维特征（d ≤ 50）上 EM > BN > none，冻结 RoBERTa（d = 768）上
BN > EM > none，仅凭「EM 残留漂移趋于 0」无法解释该反转。

假设：EM 的类条件校正是带噪估计，
        x' = x + λ·a·(γ − π)，a = dmu_tr − dmu_te
      γ（伪标签后验）出错会把样本推向错误方向、注入方差；d 越大，a 的估计
      噪声越大（均值估计误差 ~ d/n）。故净收益 = 消除的均值漂移 − 注入的
      伪标签噪声，最优 λ* 应为 (0,1) 内点，且随维度 / 伪标签错误率下降。

统一族 TTA-λ：
        Step1（批级）   Bc = B + (mu_tr − mean(B))        λ=0 即批中心对齐
        Step2（条件级） 迭代 T 次：γ = P(AI|x)；a = dmu_tr − (mu1_te − mu0_te)
                        Bc += λ·a·(γ − π)
        λ=0 → 批对齐（BNc）；λ=1 → 完全类条件对齐（EM）
        oracle 版本 γ 用真值，作为上界。

检验：
  1. F1(λ) 是否存在内点最优 λ*，且 λ* 随 d 增大而减小
  2. Ξ_eff 残留比随 λ 的变化是否满足 ρ(λ) ≈ (1 − λ(1−2e))²（e 为伪标签错误率）
  3. oracle-γ 是否显著优于预测 γ，以判定改进空间在 E 步还是对齐公式
"""
from __future__ import annotations
import io, os, sys, json, time, argparse, importlib.util, hashlib
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

SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5
N_CG = 800
CACHE = HERE / '.cache'
OUT_PATH = HERE / 'results' / 'E28_tta_lambda.json'


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


# ---------------------------------------------------------------- 特征/缓存
class CachedRoberta:
    """冻结 RoBERTa 表征，按 (llm, role, max_len) 缓存到 .cache"""

    def __init__(self, max_len=256, batch=16):
        self.name = 'roberta_frozen'
        self.max_len, self.batch = max_len, batch
        self._impl = None

    def _get_impl(self):
        if self._impl is None:
            self._impl = E27.RobertaFeat(max_len=self.max_len, batch=self.batch).fit([])
        return self._impl

    def embed(self, key, texts):
        CACHE.mkdir(parents=True, exist_ok=True)
        p = CACHE / ('emb_%s_%d.npy' % (key, self.max_len))
        if p.exists():
            try:
                a = np.load(p)
                if a.shape[0] == len(texts):
                    log('  cache hit %s (%d,%d)' % (p.name, a.shape[0], a.shape[1]))
                    return a
            except Exception:
                pass
        log('  embedding %s  n=%d ...' % (key, len(texts)))
        a = np.asarray(self._get_impl().transform(texts), dtype=np.float64)
        np.save(p, a)
        log('  saved %s' % p.name)
        return a

    def fit(self, texts):
        return self

    def transform(self, texts):
        k = hashlib.md5(('\n'.join(texts)).encode('utf-8')).hexdigest()[:12]
        return self.embed(k, texts)


FEATURES = {
    '15D': lambda: E26.Feat15D(),
    'char_2_4': lambda: E26.TfidfFeat('char', (2, 4), 30000, 5),
    'word_1_3': lambda: E26.TfidfFeat('word', (1, 3), 5000, 3),
    'roberta': lambda: CachedRoberta(max_len=256, batch=16),
}
PI = 0.5
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------- TTA 族
def tta_none(B, mu_tr, sd_tr):
    return B


def tta_bn(B, mu_tr, sd_tr):
    mu = B.mean(axis=0)
    sd = B.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (B - mu) / sd * sd_tr + mu_tr


def _bnc(B, mu_tr):
    return B + (mu_tr - B.mean(axis=0))


def tta_lambda(B, clf, mu_tr, mu0_tr, mu1_tr, lam, iters=3, oracle_g=None):
    """
    Step1 批中心对齐 → Step2 λ 加权的类条件校正（迭代 T 次）
    oracle_g: 若给定真实标签 0/1，则 E 步用真值（上界）
    """
    Bc = _bnc(B, mu_tr)
    dmu_tr = mu1_tr - mu0_tr
    nrm_off = []
    for it in range(iters):
        if oracle_g is not None:
            g = oracle_g.astype(np.float64)
        else:
            g = clf.predict_proba(Bc)[:, 1]
        w1 = g.sum()
        if w1 < 1e-6:
            break
        mu1 = (Bc * g[:, None]).sum(axis=0) / w1
        w0 = (1.0 - g).sum()
        mu0 = (Bc * (1.0 - g)[:, None]).sum(axis=0) / max(w0, 1e-9)
        a = dmu_tr - (mu1 - mu0)
        nrm_off.append(float(np.linalg.norm(a)))
        step = lam * a * (g - PI)[:, None]
        # 锁住批中心：只保留差分分量，避免 λ 步把整批平移（否则 λ 只影响收敛速度）
        step = step - step.mean(axis=0)
        Bc = Bc + step
        if oracle_g is not None:
            break          # oracle 无迭代必要
    return Bc, (nrm_off[-1] if nrm_off else 0.0)


def eval_cell(yte, pred, pb, clf):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        auc = float(roc_auc_score(yte, pb))
    return {'F1_macro': round(float(f1_score(yte, pred, average='macro')), 4),
            'AUC': round(auc, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='deepseek,glm')
    ap.add_argument('--mix', default='0')
    ap.add_argument('--feats', default='15D,char_2_4')
    ap.add_argument('--lambdas', default=','.join(str(x) for x in LAMBDAS))
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--oracle', action='store_true')
    ap.add_argument('--out', default=str(OUT_PATH))
    args = ap.parse_args()
    OUT = Path(args.out)
    targets = args.targets.split(',')
    alphas = [float(x) for x in args.mix.split(',')]
    lams = [float(x) for x in args.lambdas.split(',')]

    t0 = time.time()
    hc3_idx = E17.build_hc3_index()
    data = {}
    for m in ['chatgpt', 'deepseek'] + targets:
        hs, ais, _ = E17.build_pairs(m, hc3_idx)
        data[m] = {'human': hs, 'ai': ais}
        log('  %-10s n=%d' % (m, len(ais)))

    cg, ds = data['chatgpt'], data['deepseek']
    out = {'protocol': ('训练 ChatGPT-only(+α 混合); 测试目标 LLM 一整批无标签文本; '
                        'TTA 统一族: none / bn / bnc(λ=0) / λ∈%s / oracle-γ; '
                        '5-fold×5seeds; macro-F1' % args.lambdas),
           'lambdas': lams, 'iters': args.iters, 'feature_sets': {}}

    for fs_name in args.feats.split(','):
        if fs_name not in FEATURES:
            continue
        log('── %s ──' % fs_name)
        fs_out = {}
        for alpha in alphas:
            n_ds = 0 if alpha <= 0 else int(round(N_CG * alpha / (1 - alpha)))
            tr_h = cg['human'][:N_CG] + ds['human'][:n_ds]
            tr_a = cg['ai'][:N_CG] + ds['ai'][:n_ds]
            tr_texts = list(tr_h) + list(tr_a)
            ytr = np.array([0] * len(tr_h) + [1] * len(tr_a))

            f = FEATURES[fs_name]()
            if isinstance(f, CachedRoberta):
                A = f.embed('train_a%g' % alpha, tr_texts)
            else:
                A = np.asarray(f.fit(tr_texts).transform(tr_texts), dtype=np.float64)
            A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
            sc = StandardScaler().fit(A)
            A = sc.transform(A)
            mu_tr = A.mean(axis=0)
            sd_tr = A.std(axis=0)
            mu0_tr, mu1_tr = A[ytr == 0].mean(axis=0), A[ytr == 1].mean(axis=0)
            clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0).fit(A, ytr)
            log('  train acc=%.4f  dim=%d' % (clf.score(A, ytr), A.shape[1]))

            a_out = {'n_ds': n_ds, 'dim': int(A.shape[1])}
            for tgt in targets:
                th, ta = data[tgt]['human'], data[tgt]['ai']
                if tgt == 'deepseek' and n_ds > 0:
                    th, ta = th[n_ds:], ta[n_ds:]
                te_texts = list(th) + list(ta)
                yte = np.array([0] * len(th) + [1] * len(ta))
                if isinstance(f, CachedRoberta):
                    B0 = f.embed('tgt_%s' % tgt, te_texts)
                else:
                    B0 = np.asarray(f.transform(te_texts), dtype=np.float64)
                B0 = np.nan_to_num(B0, nan=0.0, posinf=0.0, neginf=0.0)
                B0 = sc.transform(B0)

                cell = {}
                variants = [('none', lambda B: (B, 0.0)),
                            ('bn', lambda B: (tta_bn(B, mu_tr, sd_tr), 0.0))]
                for lam in lams:
                    variants.append(('lam_%g' % lam,
                                     (lambda L: (lambda B: tta_lambda(B, clf, mu_tr, mu0_tr, mu1_tr, L, args.iters)))(lam)))
                if args.oracle:
                    variants.append(('oracle', lambda B: tta_lambda(B, clf, mu_tr, mu0_tr, mu1_tr, 1.0, 1, oracle_g=yte)))

                for vname, fn in variants:
                    B, na = fn(B0)
                    xi = estimate_Xi(A, ytr, B[yte == 1], B[yte == 0])['xi_eff']
                    p = clf.predict(B)
                    pb = clf.predict_proba(B)[:, 1]
                    per_fold = []
                    for seed in SEEDS:
                        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                        for _, idx in skf.split(np.zeros(len(yte)), yte):
                            per_fold.append(float(f1_score(yte[idx], p[idx], average='macro')))
                    cell[vname] = {'F1_macro': round(float(np.mean(per_fold)), 4),
                                   'std': round(float(np.std(per_fold)), 4),
                                   'Xi_eff': round(float(xi), 4),
                                   'norm_a': round(float(na), 4),
                                   'per_fold_F1': [round(x, 4) for x in per_fold]}
                base = np.array(cell['none']['per_fold_F1'])
                for v in list(cell):
                    if v == 'none':
                        continue
                    cur = np.array(cell[v]['per_fold_F1'])
                    t_s, p_s = stats.ttest_rel(cur, base)
                    cell[v]['delta'] = round(float(np.mean(cur - base)), 4)
                    cell[v]['p_paired_t'] = float(p_s)
                x0 = cell['none']['Xi_eff']
                cell['resid_ratio'] = {v: (round(cell[v]['Xi_eff'] / x0, 4) if abs(x0) > 1e-6 else None)
                                       for v in cell if v not in ('resid_ratio',)}
                # 伪标签错误率（事后分析用）
                cell['pseudo_err'] = round(1.0 - float((clf.predict(B0) == yte).mean()), 4)
                a_out[tgt] = cell
                best = max(((v, cell[v]['F1_macro']) for v in cell
                            if isinstance(cell[v], dict) and 'F1_macro' in cell[v]), key=lambda x: x[1])
                log('   α=%.2f %-9s best=%s(%.4f) none=%.4f bn=%.4f lam1=%.4f Ξ0=%.3f (%.0fs)'
                    % (alpha, tgt, best[0], best[1], cell['none']['F1_macro'],
                       cell['bn']['F1_macro'],
                       cell.get('lam_1', {}).get('F1_macro', float('nan')),
                       x0, time.time() - t0))
            fs_out['alpha_%g' % alpha] = a_out
        out['feature_sets'][fs_name] = fs_out

    out['runtime_sec'] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print('')
    print('=' * 110)
    print('E28：TTA 统一族 λ 扫描   α=%s' % args.mix)
    print('=' * 110)
    for fs_name in args.feats.split(','):
        fs = out['feature_sets'].get(fs_name, {})
        for alpha in alphas:
            a = fs.get('alpha_%g' % alpha)
            if not a:
                continue
            keys = [k for k in a[targets[0]] if isinstance(a[targets[0]][k], dict) and 'F1_macro' in a[targets[0]][k]]
            print('--- %s (dim=%s) | α=%.2f ---' % (fs_name, a.get('dim'), alpha))
            print('%-9s %s' % ('target', ''.join('%13s' % k for k in keys)))
            for tgt in targets:
                print('%-9s %s' % (tgt, ''.join('%13.4f' % a[tgt][k]['F1_macro'] for k in keys)))
            print('%-9s %s' % ('Ξ_eff', ''.join(
                '%13.3f' % a[targets[0]][k]['Xi_eff'] for k in keys)))
            print('%-9s %s' % ('残留比', ''.join(
                '%13.3f' % (a[targets[0]]['resid_ratio'][k] or 0) for k in keys)))
            print('')
    print('[SAVED] %s  (%.0fs)' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
