# -*- coding: utf-8 -*-
"""
条件不变性赤字 Ξ 与跨 LLM 实测衰减的关联验证。

Ξ 取漂移在判别方向上的投影，而非全空间漂移：漂移若与判别方向正交，
总量再大也不损害分类。设
    w = Σ_W^{-1}(μ₁−μ₀)，Δ = sqrt((μ₁−μ₀)ᵀΣ_W^{-1}(μ₁−μ₀))
    u = w/Δ，δ = μ_AI,tgt − μ_AI,src，Ξ_eff = −uᵀδ
等协方差高斯模型给出解析预测 Δ' = Δ − Ξ_eff，macro F1 ≈ 1 − Φ(−Δ'/2)。

特征集：15D / word(1,1) / word(1,2) / word(1,3) / char(2,4) / char(3,5)，
TF-IDF 一律 fit-on-train-only + TruncatedSVD(50) 以保证协方差可逆。
协议：训练 ChatGPT-only 前 800 对；in-domain 为 ChatGPT 后 200 对；
      跨 LLM 为目标 LLM 1000 对；StratifiedKFold(5)×5 seeds，macro F1。

输出：results/E19_invariance_deficit.json
"""
from __future__ import annotations
import io, os, sys, json, time, argparse, importlib.util
from pathlib import Path
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from scipy import stats
import xgboost as xgb

_spec = importlib.util.spec_from_file_location('e17', HERE / 'E17_crossllm_ghostbuster.py')
E17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E17)

SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5
SVD_DIM = 50
RIDGE = 1e-2

TARGETS = ['deepseek', 'qwen', 'glm', 'gpt-4o', 'llama3']
N_TRAIN = 800
IN_DOMAIN_HELD = 200

OUT_PATH = HERE / 'results' / 'E19_invariance_deficit.json'


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


# ────────────────── 特征集 ──────────────────
class Feat15D:
    name = '15D'

    def fit(self, texts):
        return self

    def transform(self, texts):
        return E17.feats_15d(texts)


class TfidfFeat:
    def __init__(self, analyzer, ngram, max_features, min_df, use_svd=True):
        self.analyzer = analyzer
        self.ngram = ngram
        self.max_features = max_features
        self.min_df = min_df
        self.use_svd = use_svd
        self.name = '%s_%d_%d' % (analyzer, ngram[0], ngram[1])

    def fit(self, texts):
        if self.analyzer == 'word':
            corpus = [E17.cut(t) for t in texts]
            self.vec = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram,
                                       min_df=self.min_df, max_df=0.95)
        else:
            corpus = list(texts)
            self.vec = TfidfVectorizer(analyzer='char', ngram_range=self.ngram,
                                       max_features=self.max_features, min_df=self.min_df,
                                       sublinear_tf=True)
        A = self.vec.fit_transform(corpus)
        if self.use_svd and A.shape[1] > SVD_DIM:
            self.svd = TruncatedSVD(n_components=SVD_DIM, random_state=42)
            self.svd.fit(A)
        else:
            self.svd = None
        return self

    def transform(self, texts):
        corpus = [E17.cut(t) for t in texts] if self.analyzer == 'word' else list(texts)
        A = self.vec.transform(corpus)
        if self.svd is not None:
            A = self.svd.transform(A)
        return np.asarray(A, dtype=np.float64)


FEATURE_SETS = [
    ('15D', Feat15D, {'granularity': 0, 'family': 'handcrafted'}),
    ('word_1_1', lambda: TfidfFeat('word', (1, 1), 5000, 2), {'granularity': 1, 'family': 'word'}),
    ('word_1_2', lambda: TfidfFeat('word', (1, 2), 5000, 2), {'granularity': 2, 'family': 'word'}),
    ('word_1_3', lambda: TfidfFeat('word', (1, 3), 5000, 2), {'granularity': 3, 'family': 'word'}),
    ('char_2_4', lambda: TfidfFeat('char', (2, 4), 30000, 5), {'granularity': 4, 'family': 'char'}),
    ('char_3_5', lambda: TfidfFeat('char', (3, 5), 30000, 5), {'granularity': 5, 'family': 'char'}),
]


# ────────────────── Ξ 估计 ──────────────────
def estimate_Xi(Xtr, ytr, X_te_ai, X_te_hu=None):
    """见文件头。返回 dict(xi_raw, xi_eff, Delta, Delta_prime, xi_hu_raw, xi_hu_eff)"""
    d = Xtr.shape[1]
    mu = {}
    SW = np.zeros((d, d))
    for c in (0, 1):
        Xc = Xtr[ytr == c]
        mu[c] = Xc.mean(axis=0)
        Xc0 = Xc - mu[c]
        SW += Xc0.T @ Xc0
    SW /= max(len(Xtr) - 2, 1)
    SW += RIDGE * np.trace(SW) / max(d, 1) * np.eye(d)
    SW_inv = np.linalg.pinv(SW)

    dmu = mu[1] - mu[0]
    w = SW_inv @ dmu
    Delta2 = float(dmu @ SW_inv @ dmu)
    Delta = float(np.sqrt(max(Delta2, 0.0)))
    u = w / Delta if Delta > 1e-9 else np.zeros_like(w)

    def maha(a, b):
        dv = a - b
        return float(dv @ SW_inv @ dv) / d

    d_ai = np.asarray(X_te_ai).mean(axis=0) - mu[1]
    xi_raw = maha(np.asarray(X_te_ai).mean(axis=0), mu[1])
    xi_eff = float(-u @ d_ai)
    Delta_prime = Delta + float(u @ d_ai)

    xi_hu_raw = xi_hu_eff = float('nan')
    if X_te_hu is not None and len(X_te_hu) > 1:
        d_hu = np.asarray(X_te_hu).mean(axis=0) - mu[0]
        xi_hu_raw = maha(np.asarray(X_te_hu).mean(axis=0), mu[0])
        xi_hu_eff = float(-u @ d_hu)

    return {'xi_raw': xi_raw, 'xi_eff': xi_eff, 'Delta': Delta,
            'Delta_prime': Delta_prime, 'xi_hu_raw': xi_hu_raw, 'xi_hu_eff': xi_hu_eff}


# ────────────────── 评估 ──────────────────
def eval_cell(feat, tr_texts, ytr, te_texts, yte, clf_kind='lr'):
    f = feat.fit(tr_texts)
    A = np.asarray(f.transform(tr_texts), dtype=np.float64)
    B = np.asarray(f.transform(te_texts), dtype=np.float64)
    imp = SimpleImputer(strategy='median').fit(A)
    sc = StandardScaler().fit(imp.transform(A))
    A = sc.transform(imp.transform(A))
    B = sc.transform(imp.transform(B))

    yte_a = np.asarray(yte)
    xi = estimate_Xi(A, ytr, B[yte_a == 1], B[yte_a == 0])

    fold_f1, aucs = [], []
    for seed in SEEDS:
        if clf_kind == 'lr':
            m = LogisticRegression(max_iter=2000, random_state=seed, C=1.0)
        else:
            m = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                  random_state=seed, eval_metric='logloss', verbosity=0, n_jobs=4)
        m.fit(A, np.asarray(ytr))
        p = m.predict(B)
        pb = m.predict_proba(B)[:, 1]
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for _, idx in skf.split(np.zeros(len(yte_a)), yte_a):
            fold_f1.append(float(f1_score(yte_a[idx], p[idx], average='macro')))
        aucs.append(float(roc_auc_score(yte_a, pb)))

    return {'f1': float(np.mean(fold_f1)), 'std': float(np.std(fold_f1)),
            'folds': fold_f1, 'auc': float(np.mean(aucs)), 'xi': xi}


def _reg(X, Y):
    if len(X) < 5:
        return {}
    pr, pp = stats.pearsonr(X, Y)
    sr, sp = stats.spearmanr(X, Y)
    lr = stats.linregress(X, Y)
    return {'n': int(len(X)),
            'pearson_r': round(float(pr), 4), 'pearson_p': float(pp),
            'spearman_rho': round(float(sr), 4), 'spearman_p': float(sp),
            'ols_slope': round(float(lr.slope), 6), 'ols_intercept': round(float(lr.intercept), 6),
            'ols_r2': round(float(lr.rvalue ** 2), 4), 'ols_p': float(lr.pvalue)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default=','.join(TARGETS))
    ap.add_argument('--clf', default='lr', choices=['lr', 'xgb'])
    ap.add_argument('--out', default=str(OUT_PATH))
    args = ap.parse_args()
    OUT = Path(args.out)
    targets = args.targets.split(',')

    t0 = time.time()
    log('构建 HC3 问题索引 ...')
    hc3_idx = E17.build_hc3_index()
    data = {}
    for m in ['chatgpt'] + targets:
        hs, ais, qs = E17.build_pairs(m, hc3_idx)
        data[m] = {'human': hs, 'ai': ais}
        log('  %-10s 匹配对数 = %d' % (m, len(ais)))

    cg_h, cg_a = data['chatgpt']['human'], data['chatgpt']['ai']
    tr_h, tr_a = cg_h[:N_TRAIN], cg_a[:N_TRAIN]
    id_h, id_a = cg_h[N_TRAIN:N_TRAIN + IN_DOMAIN_HELD], cg_a[N_TRAIN:N_TRAIN + IN_DOMAIN_HELD]
    tr_texts = list(tr_h) + list(tr_a)
    ytr = np.array([0] * len(tr_h) + [1] * len(tr_a))
    log('训练 ChatGPT-only=%d 对；in-domain held-out=%d 对' % (len(tr_a), len(id_a)))

    out = {
        'protocol': ('train=ChatGPT-only(%d); in-domain=ChatGPT held-out(%d); cross-LLM=1000 pairs; '
                     'StratifiedKFold(5)x5seeds=25 evals; macro-F1; clf=%s' % (len(tr_a), len(id_a), args.clf)),
        'svd_dim': SVD_DIM,
        'definitions': {
            'Xi_raw': '(1/d)*(mu_AI,tgt-mu_AI,src)^T Sigma_W^{-1} (mu_AI,tgt-mu_AI,src)  [全空间漂移]',
            'Xi_eff': '-u^T delta, u=Sigma_W^{-1}(mu1-mu0)/Delta  [判别方向投影漂移，有害为正]',
            'Delta': '源域马氏分离度 sqrt((mu1-mu0)^T Sigma_W^{-1} (mu1-mu0))',
            'Delta_prime': 'Delta + u^T delta = Delta - Xi_eff  [跨LLM后有效分离度]',
            'F1_analytic': '1 - Phi(-Delta_prime/2)  [等协方差高斯最优错误率的解析预测]',
        },
        'feature_sets': {},
    }

    rows = []
    for fs_name, ctor, meta in FEATURE_SETS:
        fs_out = {'granularity': meta['granularity'], 'family': meta['family'], 'targets': {}}
        log('── %s ──' % fs_name)

        te_texts = list(id_h) + list(id_a)
        yte = np.array([0] * len(id_h) + [1] * len(id_a))
        r_id = eval_cell(ctor(), tr_texts, ytr, te_texts, yte, args.clf)
        fs_out['in_domain'] = {'F1_macro': round(r_id['f1'], 4), 'std': round(r_id['std'], 4),
                               'AUC': round(r_id['auc'], 4),
                               'Delta': round(r_id['xi']['Delta'], 4),
                               'Xi_eff': round(r_id['xi']['xi_eff'], 4),
                               'Xi_raw': round(r_id['xi']['xi_raw'], 4)}
        log('   in-domain  F1=%.4f  Δ=%.3f  Ξ_eff=%.3f' %
            (r_id['f1'], r_id['xi']['Delta'], r_id['xi']['xi_eff']))

        for tgt in targets:
            th, ta = data[tgt]['human'], data[tgt]['ai']
            te_texts = list(th) + list(ta)
            yte = np.array([0] * len(th) + [1] * len(ta))
            r = eval_cell(ctor(), tr_texts, ytr, te_texts, yte, args.clf)
            xi = r['xi']
            pred_f1 = 1.0 - stats.norm.cdf(-xi['Delta_prime'] / 2.0)
            fs_out['targets'][tgt] = {
                'F1_macro': round(r['f1'], 4), 'std': round(r['std'], 4), 'AUC': round(r['auc'], 4),
                'Xi_raw': round(xi['xi_raw'], 4), 'Xi_eff': round(xi['xi_eff'], 4),
                'Delta': round(xi['Delta'], 4), 'Delta_prime': round(xi['Delta_prime'], 4),
                'F1_analytic': round(float(pred_f1), 4),
                'F1_gap_pred_minus_obs': round(float(pred_f1 - r['f1']), 4),
                'Delta_F1_vs_indomain': round(r_id['f1'] - r['f1'], 4),
                'Xi_hu_eff_sanity': round(xi['xi_hu_eff'], 4),
            }
            rows.append({'fs': fs_name, 'gran': meta['granularity'], 'family': meta['family'],
                         'tgt': tgt, 'Xi_raw': xi['xi_raw'], 'Xi_eff': xi['xi_eff'],
                         'Delta': xi['Delta'], 'Delta_prime': xi['Delta_prime'],
                         'F1': r['f1'], 'F1_pred': float(pred_f1),
                         'dF1': r_id['f1'] - r['f1'], 'F1_id': r_id['f1']})
            log('   %-9s F1=%.4f  Ξ_eff=%+.3f  Δ=%.3f→%.3f  解析F1=%.4f (%.0fs)'
                % (tgt, r['f1'], xi['xi_eff'], xi['Delta'], xi['Delta_prime'],
                   pred_f1, time.time() - t0))
        out['feature_sets'][fs_name] = fs_out

    # ── 关联分析 ──
    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    ok = np.isfinite(col('Xi_eff')) & np.isfinite(col('dF1'))
    out['regression'] = {
        'Xi_eff_vs_DeltaF1': _reg(col('Xi_eff')[ok], col('dF1')[ok]),
        'Xi_raw_vs_DeltaF1': _reg(col('Xi_raw')[ok], col('dF1')[ok]),
        'Delta_prime_vs_F1': _reg(col('Delta_prime')[ok], col('F1')[ok]),
        'analyticF1_vs_obsF1': _reg(col('F1_pred')[ok], col('F1')[ok]),
    }
    # 解析预测的绝对偏差
    gap = col('F1_pred')[ok] - col('F1')[ok]
    out['analytic_prediction_error'] = {
        'MAE': round(float(np.mean(np.abs(gap))), 4),
        'RMSE': round(float(np.sqrt(np.mean(gap ** 2))), 4),
        'bias': round(float(np.mean(gap)), 4),
    }

    # ── 粒度律（族内单调性）──
    mono = {}
    for fam in ['word', 'char']:
        sub = [r for r in rows if r['family'] == fam]
        if not sub:
            continue
        gs = sorted(set(r['gran'] for r in sub))
        mono[fam] = [{'gran': g,
                      'Xi_eff_mean': round(float(np.mean([r['Xi_eff'] for r in sub if r['gran'] == g])), 4),
                      'dF1_mean': round(float(np.mean([r['dF1'] for r in sub if r['gran'] == g])), 4),
                      'F1_id_mean': round(float(np.mean([r['F1_id'] for r in sub if r['gran'] == g])), 4)}
                     for g in gs]
    out['granularity_law'] = mono
    out['rows'] = rows
    out['runtime_sec'] = round(time.time() - t0, 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ── 控制台 ──
    print('')
    print('=' * 112)
    print('E19：Ξ_eff（判别方向投影漂移） vs 跨 LLM 衰减   [clf=%s]' % args.clf.upper())
    print('=' * 112)
    print('%-10s %5s %9s %8s | %s' % ('featset', 'gran', 'in-domF1', 'Δ(src)',
                                      ''.join('%18s' % t for t in targets)))
    for fs_name, _, meta in FEATURE_SETS:
        fo = out['feature_sets'][fs_name]
        row = '%-10s %5d %9.4f %8.3f | ' % (fs_name, meta['granularity'],
                                            fo['in_domain']['F1_macro'], fo['in_domain']['Delta'])
        for t in targets:
            d = fo['targets'][t]
            row += '%18s' % ('%.4f(Ξ%+0.2f)' % (d['F1_macro'], d['Xi_eff']))
        print(row)
    print('')
    print('%-10s %5s %9s %8s | %s' % ('', '', 'Δ(F1)', '', ''.join('%18s' % t for t in targets)))
    for fs_name, _, meta in FEATURE_SETS:
        fo = out['feature_sets'][fs_name]
        row = '%-10s %5d %9s %8s | ' % (fs_name, meta['granularity'], '', '')
        for t in targets:
            row += '%18s' % ('%.4f' % fo['targets'][t]['Delta_F1_vs_indomain'])
        print(row)
    print('')
    R = out['regression']
    for k in ['Xi_eff_vs_DeltaF1', 'Xi_raw_vs_DeltaF1', 'Delta_prime_vs_F1', 'analyticF1_vs_obsF1']:
        d = R.get(k, {})
        if d:
            print('%-22s Pearson r=%+.4f (p=%.3g)  Spearman ρ=%+.4f (p=%.3g)  R²=%.4f'
                  % (k, d['pearson_r'], d['pearson_p'], d['spearman_rho'], d['spearman_p'], d['ols_r2']))
    print('解析预测误差: MAE=%.4f  RMSE=%.4f  bias=%+.4f'
          % (out['analytic_prediction_error']['MAE'], out['analytic_prediction_error']['RMSE'],
             out['analytic_prediction_error']['bias']))
    print('')
    print('粒度律（族内均值）:')
    for fam, lst in mono.items():
        print('  %-5s ' % fam + ' → '.join('g%d: Ξ%+0.3f ΔF1=%.3f indom=%.3f'
                                           % (x['gran'], x['Xi_eff_mean'], x['dF1_mean'], x['F1_id_mean'])
                                           for x in lst))
    print('[SAVED] %s  (%.0fs)' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
