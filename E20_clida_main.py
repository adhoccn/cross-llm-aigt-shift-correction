# -*- coding: utf-8 -*-
"""
CLIDA（跨 LLM 不变判别分析）主实验。

S_E 需先归一化到与 S_W 同 trace 尺度，否则 γS_E 在总量中占比过低，γ 失去
相对权重含义。

考察点为混合数据效率：CLIDA 能否用 α=5% 的混合数据达到 raw 在 α=20% 的
泛化水平。若能，等价于把异源数据需求降低 4 倍。

方法：raw_lr / raw_xgb（把投影与分类器两个因素解耦）/ diag_heur /
      diag_fisher / clida / clida_xgb
环境：n_ds>0 时 env = LLM × 领域；n_ds=0 时退化为 env = 领域（代理环境）。
机制验证：变换后空间的 Ξ_eff'，检验 CLIDA 是否压低了漂移与判别方向的对齐度。
"""
from __future__ import annotations
import io, os, sys, json, time, argparse, importlib.util, collections
from pathlib import Path
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core.invariance import CLIDA, diag_fisher, diag_heuristic, estimate_Xi, scatter_matrices

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
N_CG = 800
OUT_PATH = HERE / 'results' / 'E20_clida_main.json'


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def build_domain_index():
    idx = {}
    with E17.HC3_PATH.open('r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            q = (d.get('question') or '').strip()
            if q:
                idx[q] = d.get('source') or 'unknown'
    return idx


def build_pairs_with_domain(model, hc3_idx, dom_idx):
    rows = E17.load_collected(model)
    hs, ais, doms = [], [], []
    for r in rows:
        q = (r.get('question') or '').strip()
        if q in hc3_idx:
            hs.append(hc3_idx[q]); ais.append(r['answer']); doms.append(dom_idx.get(q, 'unknown'))
    return hs, ais, doms


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
        if self.svd is not None:
            A = self.svd.transform(A)
        return np.asarray(A, dtype=np.float64)


FEATURE_SETS = [
    ('15D', lambda: Feat15D()),
    ('char_2_4', lambda: TfidfFeat('char', (2, 4), 30000, 5)),
    ('word_1_3', lambda: TfidfFeat('word', (1, 3), 5000, 2)),
]

METHODS = ['raw_lr', 'raw_xgb', 'diag_heur', 'diag_fisher', 'clida', 'clida_xgb']


def apply_method(method, Xtr, ytr, env_tr, Xte, gamma, seed):
    ytr = np.asarray(ytr)
    if method == 'raw_lr':
        T, clf_kind = (lambda X: X), 'lr'
    elif method == 'raw_xgb':
        T, clf_kind = (lambda X: X), 'xgb'
    elif method == 'diag_heur':
        M = diag_heuristic(Xtr, ytr, env_tr)
        T, clf_kind = (lambda X: X * M), 'lr'
    elif method == 'diag_fisher':
        u = diag_fisher(Xtr, ytr, env_tr, gamma=gamma)
        T, clf_kind = (lambda X: X * u), 'lr'
    elif method == 'clida':
        m = CLIDA(gamma=gamma).fit(Xtr, ytr, env_tr)
        T, clf_kind = (lambda X: m.transform(X)), 'lr'
    elif method == 'clida_xgb':
        m = CLIDA(gamma=gamma).fit(Xtr, ytr, env_tr)
        T, clf_kind = (lambda X: m.transform(X)), 'xgb'
    else:
        raise ValueError(method)
    A, B = T(Xtr), T(Xte)
    if clf_kind == 'lr':
        clf = LogisticRegression(max_iter=2000, random_state=seed, C=1.0)
    else:
        clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                random_state=seed, eval_metric='logloss', verbosity=0,
                                tree_method='hist', n_jobs=4)
    clf.fit(A, ytr)
    return clf.predict(B), clf.predict_proba(B)[:, 1], {'A': A, 'B': B}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='deepseek,glm')
    ap.add_argument('--mix', default='0,0.05,0.2', help='DeepSeek 混合比例 α（占训练总量）')
    ap.add_argument('--gamma', default='0,1,3')
    ap.add_argument('--feats', default='15D,char_2_4,word_1_3')
    ap.add_argument('--out', default=str(OUT_PATH))
    args = ap.parse_args()
    OUT = Path(args.out)
    targets = args.targets.split(',')
    alphas = [float(x) for x in args.mix.split(',')]
    gammas = [float(g) for g in args.gamma.split(',')]
    want_feats = args.feats.split(',')

    t0 = time.time()
    log('构建索引 ...')
    hc3_idx = E17.build_hc3_index()
    dom_idx = build_domain_index()
    data = {}
    for m in ['chatgpt', 'deepseek'] + targets:
        hs, ais, doms = build_pairs_with_domain(m, hc3_idx, dom_idx)
        data[m] = {'human': hs, 'ai': ais, 'dom': doms}
        log('  %-10s n=%d' % (m, len(ais)))

    cg, ds = data['chatgpt'], data['deepseek']
    out = {'protocol': ('N_CG=%d; α = DeepSeek 占训练总量比例; n_ds = int(N_CG*α/(1-α)); '
                        'env = LLM×domain (n_ds>0) 或 domain (n_ds=0); '
                        'test = target 1000 pairs (deepseek 排除前 n_ds); 5-fold×5seeds; macro-F1'
                        % N_CG),
           'S_E_normalized': True, 'feature_sets': {}}

    for fs_name, ctor in FEATURE_SETS:
        if fs_name not in want_feats:
            continue
        fs_out = {}
        log('── 特征集 %s ──' % fs_name)
        for alpha in alphas:
            n_ds = 0 if alpha <= 0 else int(round(N_CG * alpha / (1 - alpha)))
            a_out = {'n_ds': n_ds}
            # 训练集
            tr_h = cg['human'][:N_CG] + ds['human'][:n_ds]
            tr_a = cg['ai'][:N_CG] + ds['ai'][:n_ds]
            tr_d = cg['dom'][:N_CG] + ds['dom'][:n_ds]
            llm_h = np.concatenate([np.zeros(N_CG), np.ones(n_ds)])
            tr_texts = list(tr_h) + list(tr_a)
            ytr = np.array([0] * len(tr_h) + [1] * len(tr_a))
            dom_full = list(tr_d) + list(tr_d)
            llm_full = np.concatenate([llm_h, llm_h])
            env_tr = (np.array(['%d|%s' % (a, b) for a, b in zip(llm_full, dom_full)])
                      if n_ds > 0 else np.array(['%s' % b for b in dom_full]))
            assert len(env_tr) == len(tr_texts)

            f = ctor().fit(tr_texts)
            A = np.asarray(f.transform(tr_texts), dtype=np.float64)
            imp = SimpleImputer(strategy='median').fit(A)
            sc = StandardScaler().fit(imp.transform(A))
            A = sc.transform(imp.transform(A))
            S_B, S_W, S_E = scatter_matrices(A, ytr, env_tr)
            a_out['scatter'] = {'tr_S_B': round(float(np.trace(S_B)), 4),
                                'tr_S_W': round(float(np.trace(S_W)), 4),
                                'tr_S_E': round(float(np.trace(S_E)), 6),
                                'ratio_SE_SW': round(float(np.trace(S_E) / max(np.trace(S_W), 1e-12)), 6)}

            # 测试集特征缓存
            Bmap = {}
            for tgt in targets:
                th, ta = data[tgt]['human'], data[tgt]['ai']
                if tgt == 'deepseek' and n_ds > 0:
                    th, ta = th[n_ds:], ta[n_ds:]
                te_texts = list(th) + list(ta)
                yte = np.array([0] * len(th) + [1] * len(ta))
                Bmap[tgt] = (sc.transform(imp.transform(np.asarray(f.transform(te_texts), dtype=np.float64))), yte)

            for gamma in gammas:
                g_out = {}
                for tgt in targets:
                    B, yte = Bmap[tgt]
                    cell = {}
                    for method in METHODS:
                        fold_f1, aucs, xi_list = [], [], []
                        p42 = None
                        for seed in SEEDS:
                            p, pb, info = apply_method(method, A, ytr, env_tr, B, gamma, seed)
                            if seed == 42:
                                p42 = (yte.copy(), p.copy())
                                xi_list.append(estimate_Xi(info['A'], ytr,
                                                           info['B'][yte == 1], info['B'][yte == 0])['xi_eff'])
                            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                            for _, idx in skf.split(np.zeros(len(yte)), yte):
                                fold_f1.append(float(f1_score(yte[idx], p[idx], average='macro')))
                            aucs.append(float(roc_auc_score(yte, pb)))
                        cell[method] = {'F1_macro': round(float(np.mean(fold_f1)), 4),
                                        'std': round(float(np.std(fold_f1)), 4),
                                        'AUC': round(float(np.mean(aucs)), 4),
                                        'per_fold_F1': [round(x, 4) for x in fold_f1],
                                        'Xi_eff_after': round(float(np.mean(xi_list)), 4) if xi_list else None,
                                        '_p42': p42}
                        log('   α=%.2f γ=%-4g %-9s %-11s F1=%.4f Ξ\'=%s (%.0fs)'
                            % (alpha, gamma, tgt, method, np.mean(fold_f1),
                               ('%+.3f' % np.mean(xi_list)) if xi_list else 'n/a', time.time() - t0))
                    # 显著性 vs raw_lr（配对 t，25 folds）
                    base = np.array(cell['raw_lr']['per_fold_F1'])
                    sig = {}
                    for m2 in METHODS[1:]:
                        cur = np.array(cell[m2]['per_fold_F1'])
                        t_s, p_s = stats.ttest_rel(cur, base)
                        yb, p1 = cell['raw_lr']['_p42']
                        _, p2 = cell[m2]['_p42']
                        c1, c2q = (p1 == yb), (p2 == yb)
                        b_ = int(np.sum(c1 & ~c2q)); c_ = int(np.sum(~c1 & c2q))
                        mn_p = 1.0
                        if b_ + c_ > 0:
                            from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
                            mn_p = float(sm_mcnemar(np.array([[0, b_], [c_, 0]]) + 1, exact=True).pvalue)
                        sig[m2] = {'delta_vs_raw_lr': round(float(np.mean(cur - base)), 4),
                                   'p_paired_t': float(p_s), 'mcnemar_p_seed42': mn_p}
                    cell['_sig_vs_raw_lr'] = sig
                    g_out[tgt] = cell
                a_out['gamma_%g' % gamma] = g_out
            fs_out['alpha_%g' % alpha] = a_out
        out['feature_sets'][fs_name] = fs_out

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k != '_p42'}
        return o

    out = clean(out)
    out['runtime_sec'] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print('')
    print('=' * 128)
    print('E20 v2：CLIDA（macro F1；Ξ\' = 变换后空间的 Ξ_eff）')
    print('=' * 128)
    for fs_name in want_feats:
        fs = out['feature_sets'].get(fs_name, {})
        for alpha in alphas:
            a = fs.get('alpha_%g' % alpha)
            if not a:
                continue
            print('--- %s | α=%.2f (n_ds=%d) | tr(S_E)/tr(S_W)=%.5f ---'
                  % (fs_name, alpha, a['n_ds'], a['scatter']['ratio_SE_SW']))
            print('%-9s %-5s %s' % ('target', 'γ', ''.join('%17s' % m for m in METHODS)))
            for gamma in gammas:
                g = a['gamma_%g' % gamma]
                for tgt in targets:
                    row = '%-9s %-5g ' % (tgt, gamma)
                    for m in METHODS:
                        d = g[tgt][m]
                        row += '%17s' % ('%.4f(Ξ%+.2f)' % (d['F1_macro'], d['Xi_eff_after'] or 0))
                    print(row)
            print('')
    print('[SAVED] %s  (%.0fs)' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
