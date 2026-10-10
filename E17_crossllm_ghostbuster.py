# -*- coding: utf-8 -*-
"""
跨 LLM 场景下 GHOSTBUSTER 类 TF-IDF 特征与本文 15D 的同协议对照。

协议：
  训练：ChatGPT-only 全量 1000 对 / Mixed-20%（80% ChatGPT + 20% DeepSeek，
        问题级隔离）
  测试：目标 LLM 1000 对，StratifiedKFold(5, seed) 每折独立评估
  重复：seeds = {42, 123, 456, 789, 2024}，每 cell 25 次评估
  指标：macro F1，附带 binary F1 / AUC

方法：
  15D+XGBoost         n_estimators=100, max_depth=4
  TF-IDF 词级 + LR    jieba, 1-3gram, 5K, min_df=2, max_df=0.95
  TF-IDF 词级 + MLP   (256,128)
  TF-IDF 字符级 + LR  2-4gram, 30K, min_df=5, sublinear_tf
  TF-IDF 字符级 + XGB n_estimators=200, max_depth=4
  全部严格 fit-on-train-only（词表 / IDF / scaler 只由训练折估计）

统计：15D vs 各 TF-IDF 的 25 次 fold 配对差 → 配对 t 检验 + bootstrap CI；
      另在 seed=42 的全测试集预测上做 McNemar 精确检验。

输出：results/E17_crossllm_ghostbuster.json
"""
from __future__ import annotations
import io, os, sys, json, time, argparse
from pathlib import Path
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core.feature_extractor_15d import extract_features_15d

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from scipy import stats
import xgboost as xgb
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5
DATA_DIR = Path(os.environ.get('TTA_DATA_DIR', HERE / 'data'))
HC3_PATH = DATA_DIR / 'all.jsonl'
COLLECTED_DIR = DATA_DIR / 'collected'
OUT_PATH = HERE / 'results' / 'E17_crossllm_ghostbuster.json'


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


# ---------------- 数据 ----------------
def build_hc3_index():
    idx = {}
    with HC3_PATH.open('r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            q = (row.get('question') or '').strip()
            hs = row.get('human_answers') or []
            if q and hs:
                idx[q] = hs[0]
    return idx


def load_collected(model_name):
    fp = COLLECTED_DIR / ('%s_seed42_n1000.jsonl' % model_name)
    rows = []
    with fp.open('r', encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_pairs(model_name, hc3_idx):
    """返回 (human_texts, ai_texts, questions)，按 collected 中的问题匹配人类答案"""
    rows = load_collected(model_name)
    hs, ais, qs = [], [], []
    for r in rows:
        q = (r.get('question') or '').strip()
        if q in hc3_idx:
            hs.append(hc3_idx[q])
            ais.append(r['answer'])
            qs.append(q)
    return hs, ais, qs


# ---------------- 特征 / 模型 ----------------
_F15_CACHE = {}


def feats_15d(texts):
    """带缓存的 15D 特征提取（跨 seed 复用，避免重复计算）"""
    out = []
    for t in texts:
        v = _F15_CACHE.get(t)
        if v is None:
            try:
                v = extract_features_15d(t)
            except Exception:
                v = [np.nan] * 15
            _F15_CACHE[t] = v
        out.append(v)
    return np.array(out, dtype=np.float64)


_CUT_CACHE = {}


def cut(t):
    try:
        import jieba
        v = _CUT_CACHE.get(t)
        if v is None:
            v = ' '.join(jieba.lcut(t))
            _CUT_CACHE[t] = v
        return v
    except Exception:
        return ' '.join(list(t))


class WordTFIDF:
    name = 'tfidf_word'

    def fit(self, texts):
        self.vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=2, max_df=0.95)
        self.vec.fit([cut(t) for t in texts])
        return self

    def transform(self, texts):
        return self.vec.transform([cut(t) for t in texts])


class CharTFIDF:
    name = 'tfidf_char'

    def fit(self, texts):
        self.vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), min_df=5,
                                   max_features=30000, sublinear_tf=True)
        self.vec.fit(texts)
        return self

    def transform(self, texts):
        return self.vec.transform(texts)


def mk_xgb_15d(seed):
    # 与表 6（run_unified_5fold_5seed_v2.py）保持完全一致：默认 exact 分裂
    return xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                             random_state=seed, eval_metric='logloss', verbosity=0,
                             n_jobs=4)


def mk_xgb_tfidf(seed):
    return xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, eval_metric='logloss', verbosity=0,
                             tree_method='hist', n_jobs=4)


def train_eval_15d(tr_texts, ytr, te_texts, yte, seed):
    Xtr = feats_15d(tr_texts)
    Xte = feats_15d(te_texts)
    imp = SimpleImputer(strategy='median').fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    A = sc.transform(imp.transform(Xtr))
    B = sc.transform(imp.transform(Xte))
    m = mk_xgb_15d(seed)
    m.fit(A, ytr)
    return m.predict(B), m.predict_proba(B)[:, 1]


def train_eval_tfidf(vec_cls, clf_kind, tr_texts, ytr, te_texts, yte, seed):
    v = vec_cls().fit(tr_texts)
    A = v.transform(tr_texts)
    B = v.transform(te_texts)
    if clf_kind == 'lr':
        m = LogisticRegression(max_iter=2000, random_state=seed, C=1.0)
        m.fit(A, ytr)
        pb = m.predict_proba(B)[:, 1]
        return m.predict(B), pb
    if clf_kind == 'mlp':
        m = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200, random_state=seed,
                          early_stopping=True, n_iter_no_change=8)
        m.fit(A.toarray(), ytr)
        Bd = B.toarray()
        pb = m.predict_proba(Bd)[:, 1]
        return m.predict(Bd), pb
    if clf_kind == 'xgb':
        m = mk_xgb_tfidf(seed)
        m.fit(A, ytr)
        pb = m.predict_proba(B)[:, 1]
        return m.predict(B), pb
    raise ValueError(clf_kind)


METHODS = [
    ('15d_xgb', None, None),
    ('tfidf_word_lr', WordTFIDF, 'lr'),
    ('tfidf_word_mlp', WordTFIDF, 'mlp'),
    ('tfidf_char_lr', CharTFIDF, 'lr'),
    ('tfidf_char_xgb', CharTFIDF, 'xgb'),
]


def metrics(y, p, pb):
    return {
        'F1_macro': float(f1_score(y, p, average='macro')),
        'F1': float(f1_score(y, p, zero_division=0)),
        'P': float(precision_score(y, p, zero_division=0)),
        'R': float(recall_score(y, p, zero_division=0)),
        'AUC': float(roc_auc_score(y, pb)) if len(set(y)) > 1 else float('nan'),
    }


def bootstrap_ci(diffs, n=10000, seed=42):
    d = np.asarray(diffs, dtype=float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='deepseek,qwen,glm')
    ap.add_argument('--extra', action='store_true', help='追加 gpt-4o / llama3')
    ap.add_argument('--out', default=str(OUT_PATH))
    args = ap.parse_args()
    OUT_PATH = Path(args.out)
    targets = args.targets.split(',')
    if args.extra:
        targets += ['gpt-4o', 'llama3']

    t0 = time.time()
    log('构建 HC3 问题索引 ...')
    hc3_idx = build_hc3_index()
    log('  HC3 问题数 = %d' % len(hc3_idx))

    data = {}
    for m in ['chatgpt'] + targets:
        hs, ais, qs = build_pairs(m, hc3_idx)
        data[m] = {'human': hs, 'ai': ais, 'q': qs}
        log('  %-10s 匹配对数 = %d' % (m, len(ais)))

    cg_h, cg_a = data['chatgpt']['human'], data['chatgpt']['ai']
    n_cg = len(cg_a)

    # Mixed-20%：ChatGPT 前 80% + DeepSeek 前 20%（若 deepseek 为测试目标，则测试时排除前 20%）
    mix_n = int(n_cg * 0.2)
    keep = int(n_cg * 0.8)
    if 'deepseek' in data:
        ds_h, ds_a = data['deepseek']['human'], data['deepseek']['ai']
        mix_h = cg_h[:keep] + ds_h[:mix_n]
        mix_a = cg_a[:keep] + ds_a[:mix_n]
    else:
        mix_h, mix_a = cg_h[:keep], cg_a[:keep]

    # 目标测试集（Mixed-20% 配置下 deepseek 用 held-out 后段，避免训练-测试重叠）
    test_sets = {}
    for t in targets:
        h, a = data[t]['human'], data[t]['ai']
        test_sets[t] = {'full': (h, a),
                        'heldout': (h[mix_n:], a[mix_n:]) if t == 'deepseek' else (h, a)}

    out = {'protocol': 'train=ChatGPT-only / Mixed-20%; test=target LLM 1000 pairs; '
                       'StratifiedKFold(5) x seeds{42,123,456,789,2024} = 25 evals/cell; macro-F1 (表6口径)',
           'n_train_chatgpt_only': n_cg,
           'n_train_mixed20': len(mix_a),
           'mixed20_note': 'ChatGPT 前80%%(%d) + DeepSeek 前20%%(%d)；DeepSeek 作为测试目标时仅用后 %d 条，问题级隔离'
                           % (keep, mix_n, len(test_sets['deepseek']['heldout'][0]) if 'deepseek' in test_sets else 0),
           'configs': {}}

    for cfg in ['chatgpt_only', 'mixed20']:
        out['configs'][cfg] = {}
        for tgt in targets:
            if cfg == 'chatgpt_only':
                tr_h, tr_a = cg_h, cg_a
                te_h, te_a = test_sets[tgt]['full']
            else:
                tr_h, tr_a = mix_h, mix_a
                te_h, te_a = test_sets[tgt]['heldout']
            X_te_text = list(te_h) + list(te_a)
            y_te = np.array([0] * len(te_h) + [1] * len(te_a))

            cell = {}
            for mname, vcls, ckind in METHODS:
                fold_recs = []
                preds_seed42 = {}
                for seed in SEEDS:
                    tr_texts = list(tr_h) + list(tr_a)
                    ytr = np.array([0] * len(tr_h) + [1] * len(tr_a))
                    if mname == '15d_xgb':
                        p_all, pb_all = train_eval_15d(tr_texts, ytr, X_te_text, y_te, seed)
                    else:
                        p_all, pb_all = train_eval_tfidf(vcls, ckind, tr_texts, ytr, X_te_text, y_te, seed)
                    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                    per_fold = []
                    for _, te_idx in skf.split(np.zeros(len(y_te)), y_te):
                        per_fold.append(metrics(y_te[te_idx], p_all[te_idx], pb_all[te_idx]))
                    fold_recs.extend(per_fold)
                    if seed == 42:
                        preds_seed42 = {'y': y_te.tolist(), 'p': p_all.tolist()}
                agg = {k: (float(np.mean([r[k] for r in fold_recs])),
                           float(np.std([r[k] for r in fold_recs])))
                       for k in ('F1_macro', 'F1', 'P', 'R', 'AUC')}
                cell[mname] = {
                    'mean': {k: round(v[0], 4) for k, v in agg.items()},
                    'std': {k: round(v[1], 4) for k, v in agg.items()},
                    'per_fold_F1_macro': [round(r['F1_macro'], 4) for r in fold_recs],
                    '_preds_seed42': preds_seed42,
                }
                log('  %-12s %-10s %-14s macroF1=%.4f±%.4f  binF1=%.4f  AUC=%.4f  (%.0fs)'
                    % (cfg, tgt, mname, agg['F1_macro'][0], agg['F1_macro'][1],
                       agg['F1'][0], agg['AUC'][0], time.time() - t0))

            # 显著性：15D vs 各 TF-IDF（25 次 fold 配对）
            sig = {}
            base = np.array(cell['15d_xgb']['per_fold_F1_macro'])
            for mname, _, _ in METHODS[1:]:
                cur = np.array(cell[mname]['per_fold_F1_macro'])
                d = cur - base
                t_stat, p_val = stats.ttest_rel(cur, base)
                lo, hi = bootstrap_ci(d)
                # McNemar（seed=42 全测试集样本级）
                try:
                    yb = np.array(cell['15d_xgb']['_preds_seed42']['y'])
                    pb1 = np.array(cell['15d_xgb']['_preds_seed42']['p'])
                    pb2 = np.array(cell[mname]['_preds_seed42']['p'])
                    c1 = (pb1 == yb)
                    c2 = (pb2 == yb)
                    b = int(np.sum(c1 & ~c2)); c = int(np.sum(~c1 & c2))
                    tbl = [[0, b], [c, 0]]
                    if b + c > 0:
                        r = sm_mcnemar(np.array(tbl) + 1, exact=True)
                        mn_p, bn, cn = float(r.pvalue), b, c
                    else:
                        mn_p, bn, cn = 1.0, b, c
                except Exception as e:
                    mn_p, bn, cn = float('nan'), None, None
                sig[mname] = {'delta_macroF1': round(float(np.mean(d)), 4),
                              'paired_t': round(float(t_stat), 4),
                              'p_paired_t': float(p_val),
                              'boot95CI': [round(lo, 4), round(hi, 4)],
                              'mcnemar_p_seed42': mn_p,
                              'n01': bn, 'n10': cn}
            cell['_significance_vs_15d'] = sig
            out['configs'][cfg][tgt] = cell

    # 清理内部字段（保存体积）
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k != '_preds_seed42'}
        return o

    out = clean(out)
    out['runtime_sec'] = round(time.time() - t0, 1)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 控制台汇总
    print('')
    print('=' * 108)
    print('跨 LLM 场景：本文 15D+XGB vs GHOSTBUSTER 类 TF-IDF 基线（macro F1，表 6 同协议）')
    print('=' * 108)
    for cfg in ['chatgpt_only', 'mixed20']:
        print('--- %s ---' % cfg)
        hdr = '%-10s' % 'target' + ''.join('%18s' % m for m, _, _ in METHODS)
        print(hdr)
        for tgt in targets:
            row = '%-10s' % tgt
            for mname, _, _ in METHODS:
                c = out['configs'][cfg][tgt][mname]
                row += '%18s' % ('%.4f±%.4f' % (c['mean']['F1_macro'], c['std']['F1_macro']))
            print(row)
        print('')
    print('[SAVED] %s   (%.0fs)' % (OUT_PATH, time.time() - t0))


if __name__ == '__main__':
    main()
