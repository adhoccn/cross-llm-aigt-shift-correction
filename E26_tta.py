# -*- coding: utf-8 -*-
"""
测试时自适应（TTA）与漂移消除的理论上限。

场景：训练端 ChatGPT-only（α=0，跨 LLM 衰减最严重），测试端收到一整批
来自未知 LLM 的无标签文本。设源域类条件均值 μ0, μ1，目标域 AI 类漂移 δ，
human 类不漂移（人类答案与 LLM 无关），dmu = μ1 − μ0。

(1) 批对齐（TTA-BN）：用测试批次的混合均值/方差做标准化，对齐目标是混合
    中心 (μ0+μ1+δ)/2 而非类条件中心，AI 类残留漂移 δ_BN = δ/2。
(2) 类条件对齐（TTA-EM）：用分类器后验做 E 步、估计测试端类条件均值，再把
    dmu_te 对齐到 dmu_tr；高斯且两类可分离下残留 δ_EM → 0。

验证：Δ_F1（相对无 TTA）恢复量、Ξ_eff 在对齐前后的变化、与 CLIDA 叠加
（训练端不变性 + 测试端对齐）是否互补。
协议：5-fold×5 seeds，macro F1；α ∈ {0, 0.05, 0.2}；目标 LLM 全量。
"""
from __future__ import annotations
import io, os, sys, json, time, argparse, importlib.util
from pathlib import Path
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core.invariance import CLIDA, estimate_Xi

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from scipy import stats

_spec = importlib.util.spec_from_file_location('e17', HERE / 'E17_crossllm_ghostbuster.py')
E17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E17)

SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5
SVD_DIM = 50
N_CG = 800
OUT_PATH = HERE / 'results' / 'E26_tta.json'


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


class Feat15D:
    name = '15D'

    def fit(self, t):
        return self

    def transform(self, t):
        return E17.feats_15d(t)


class TfidfFeat:
    def __init__(self, analyzer, ngram, max_features, min_df):
        self.analyzer, self.ngram = analyzer, ngram
        self.max_features, self.min_df = max_features, min_df
        self.name = '%s_%d_%d' % (analyzer, ngram[0], ngram[1])

    def fit(self, texts):
        corpus = ([E17.cut(t) for t in texts] if self.analyzer == 'word' else list(texts))
        if self.analyzer == 'word':
            self.vec = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram,
                                       min_df=self.min_df, max_df=0.95)
        else:
            self.vec = TfidfVectorizer(analyzer='char', ngram_range=self.ngram,
                                       max_features=self.max_features, min_df=self.min_df,
                                       sublinear_tf=True)
        A = self.vec.fit_transform(corpus)
        self.svd = TruncatedSVD(n_components=SVD_DIM, random_state=42).fit(A) if A.shape[1] > SVD_DIM else None
        return self

    def transform(self, texts):
        corpus = ([E17.cut(t) for t in texts] if self.analyzer == 'word' else list(texts))
        A = self.vec.transform(corpus)
        return np.asarray(A, dtype=np.float64) if self.svd is None else np.asarray(self.svd.transform(A), dtype=np.float64)


FEATURE_SETS = [('15D', lambda: Feat15D()),
                ('char_2_4', lambda: TfidfFeat('char', (2, 4), 30000, 5))]

TTA_MODES = ['none', 'bn', 'em']


def tta_bn(B, mu_tr, sd_tr):
    """批标准化后仿射回训练坐标系（保留分类器尺度语义）"""
    mu = B.mean(axis=0)
    sd = B.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (B - mu) / sd
    return Z * sd_tr + mu_tr


def tta_em(B, clf, mu0_tr, mu1_tr, iters=3):
    """
    EM 式类条件对齐：
      E 步：γ_i = P(AI | x_i)（用当前特征下的分类器）
      M 步：μ_c,te = Σ γ_ic x_i / Σ γ_ic
      对齐：把 dmu_te 拉回 dmu_tr，同时对齐混合中心
    """
    dmu_tr = mu1_tr - mu0_tr
    Bc = B.copy()
    gamma = clf.predict_proba(Bc)[:, 1]
    for _ in range(iters):
        w1 = gamma
        w0 = 1.0 - gamma
        mu1_te = (Bc * w1[:, None]).sum(axis=0) / max(w1.sum(), 1e-9)
        mu0_te = (Bc * w0[:, None]).sum(axis=0) / max(w0.sum(), 1e-9)
        dmu_te = mu1_te - mu0_te
        a = dmu_tr - dmu_te                      # 需要补的方向差
        # x' = x + a*(γ-0.5)：使 dmu 对齐（γ 分离良好时 E[γ|AI]≈1, E[γ|H]≈0）
        Bc = Bc + a * (gamma - 0.5)[:, None]
        # 混合中心对齐到训练混合中心
        centre_tr = 0.5 * (mu0_tr + mu1_tr)
        centre_te = 0.5 * (mu0_te + mu1_te) + 0.5 * a * 0.0
        Bc = Bc + (centre_tr - Bc.mean(axis=0))
        gamma = clf.predict_proba(Bc)[:, 1]
    return Bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='deepseek,qwen,glm,gpt-4o,llama3')
    ap.add_argument('--mix', default='0,0.05,0.2')
    ap.add_argument('--feats', default='15D,char_2_4')
    ap.add_argument('--clida', action='store_true', help='叠加 CLIDA 训练端投影')
    ap.add_argument('--gamma', default='1.0')
    ap.add_argument('--em-iters', type=int, default=3)
    ap.add_argument('--out', default=str(OUT_PATH))
    args = ap.parse_args()
    OUT = Path(args.out)
    targets = args.targets.split(',')
    alphas = [float(x) for x in args.mix.split(',')]
    want = args.feats.split(',')
    gamma = float(args.gamma)

    t0 = time.time()
    hc3_idx = E17.build_hc3_index()
    data = {}
    for m in ['chatgpt', 'deepseek'] + targets:
        hs, ais, _ = E17.build_pairs(m, hc3_idx)
        data[m] = {'human': hs, 'ai': ais}
        log('  %-10s n=%d' % (m, len(ais)))

    cg, ds = data['chatgpt'], data['deepseek']
    out = {'protocol': ('train=ChatGPT-only 或 α 混合; test=目标 LLM 一整批无标签文本; '
                        'TTA: none / bn(批标准化) / em(EM类条件对齐); 5-fold×5seeds; macro-F1'),
           'em_iters': args.em_iters, 'clida': args.clida, 'feature_sets': {}}

    for fs_name, ctor in FEATURE_SETS:
        if fs_name not in want:
            continue
        fs_out = {}
        log('── %s ──' % fs_name)
        for alpha in alphas:
            n_ds = 0 if alpha <= 0 else int(round(N_CG * alpha / (1 - alpha)))
            tr_h = cg['human'][:N_CG] + ds['human'][:n_ds]
            tr_a = cg['ai'][:N_CG] + ds['ai'][:n_ds]
            tr_texts = list(tr_h) + list(tr_a)
            ytr = np.array([0] * len(tr_h) + [1] * len(tr_a))
            env_tr = (np.concatenate([np.zeros(N_CG), np.ones(n_ds)]).repeat(1) if False else
                      np.array([0] * N_CG + [1] * n_ds))
            env_full = np.concatenate([env_tr, env_tr])

            f = ctor().fit(tr_texts)
            A = np.asarray(f.transform(tr_texts), dtype=np.float64)
            imp = SimpleImputer(strategy='median').fit(A)
            sc = StandardScaler().fit(imp.transform(A))
            A = sc.transform(imp.transform(A))
            mu_tr = A.mean(axis=0)
            sd_tr = A.std(axis=0)
            mu0_tr, mu1_tr = A[ytr == 0].mean(axis=0), A[ytr == 1].mean(axis=0)

            clida_m = None
            if args.clida and n_ds > 0:
                clida_m = CLIDA(gamma=gamma).fit(A, ytr, env_full)
                A = clida_m.transform(A)
                mu0_tr, mu1_tr = A[ytr == 0].mean(axis=0), A[ytr == 1].mean(axis=0)
                mu_tr, sd_tr = A.mean(axis=0), A.std(axis=0)

            a_out = {'n_ds': n_ds}
            for tgt in targets:
                th, ta = data[tgt]['human'], data[tgt]['ai']
                if tgt == 'deepseek' and n_ds > 0:
                    th, ta = th[n_ds:], ta[n_ds:]
                te_texts = list(th) + list(ta)
                yte = np.array([0] * len(th) + [1] * len(ta))
                B0 = sc.transform(imp.transform(np.asarray(f.transform(te_texts), dtype=np.float64)))
                if clida_m is not None:
                    B0 = clida_m.transform(B0)

                # 训练分类器（在训练空间）
                clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0).fit(A, ytr)
                cell = {}
                xis = {}
                for mode in TTA_MODES:
                    if mode == 'none':
                        B = B0
                    elif mode == 'bn':
                        B = tta_bn(B0, mu_tr, sd_tr)
                    else:
                        B = tta_em(B0, clf, mu0_tr, mu1_tr, args.em_iters)
                    xis[mode] = estimate_Xi(A, ytr, B[yte == 1], B[yte == 0])['xi_eff']
                    p = clf.predict(B)
                    pb = clf.predict_proba(B)[:, 1]
                    fold_f1, aucs = [], []
                    for seed in SEEDS:
                        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                        for _, idx in skf.split(np.zeros(len(yte)), yte):
                            fold_f1.append(float(f1_score(yte[idx], p[idx], average='macro')))
                    aucs.append(float(roc_auc_score(yte, pb)))
                    cell[mode] = {'F1_macro': round(float(np.mean(fold_f1)), 4),
                                  'std': round(float(np.std(fold_f1)), 4),
                                  'AUC': round(float(np.mean(aucs)), 4),
                                  'Xi_eff': round(float(xis[mode]), 4),
                                  'per_fold_F1': [round(x, 4) for x in fold_f1]}
                    log('   α=%.2f %-9s %-5s F1=%.4f Ξ_eff=%+.3f (%.0fs)'
                        % (alpha, tgt, mode, np.mean(fold_f1), xis[mode], time.time() - t0))
                base = np.array(cell['none']['per_fold_F1'])
                for mode in ['bn', 'em']:
                    cur = np.array(cell[mode]['per_fold_F1'])
                    t_s, p_s = stats.ttest_rel(cur, base)
                    cell[mode]['delta_vs_none'] = round(float(np.mean(cur - base)), 4)
                    cell[mode]['p_paired_t'] = float(p_s)
                # 理论检验：bn 是否只恢复一半（δ/2）
                d_none, d_bn, d_em = xis['none'], xis['bn'], xis['em']
                cell['theory'] = {
                    'Xi_none': round(float(d_none), 4),
                    'Xi_bn': round(float(d_bn), 4),
                    'Xi_em': round(float(d_em), 4),
                    'bn_residual_ratio': round(float(d_bn / d_none), 4) if abs(d_none) > 1e-6 else None,
                    'em_residual_ratio': round(float(d_em / d_none), 4) if abs(d_none) > 1e-6 else None,
                    'note': '理论预测 bn 残留 ≈ 0.5（只消除一半漂移），em 残留 ≈ 0',
                }
                a_out[tgt] = cell
            fs_out['alpha_%g' % alpha] = a_out
        out['feature_sets'][fs_name] = fs_out

    out['runtime_sec'] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print('')
    print('=' * 116)
    print('E26：测试时自适应（TTA）  [CLIDA=%s]' % args.clida)
    print('=' * 116)
    for fs_name in want:
        fs = out['feature_sets'].get(fs_name, {})
        for alpha in alphas:
            a = fs.get('alpha_%g' % alpha)
            if not a:
                continue
            print('--- %s | α=%.2f ---' % (fs_name, alpha))
            print('%-9s %s' % ('target', ''.join('%26s' % m for m in TTA_MODES)))
            for tgt in targets:
                row = '%-9s ' % tgt
                for m in TTA_MODES:
                    d = a[tgt][m]
                    row += '%26s' % ('%.4f(Ξ%+.3f)' % (d['F1_macro'], d['Xi_eff']))
                print(row)
            print('%-9s %s' % ('残留比', ''.join('%26s' % (
                '-' if m == 'none' else '%.3f' % (a[tg]['theory']['%s_residual_ratio' % (
                    'bn' if m == 'bn' else 'em')] or 0)) for m in TTA_MODES for tg in [targets[0]])))
            print('')
    print('[SAVED] %s  (%.0fs)' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
