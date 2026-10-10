# -*- coding: utf-8 -*-
"""
模型训练与评估（对齐论文 §3.3 / §3.4 协议）
"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier


def make_model(seed=42, n_estimators=200, max_depth=4, lr=0.1):
    try:
        return XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, n_jobs=4, eval_metric='logloss', verbosity=0)
    except TypeError:
        return XGBClassifier(random_state=seed)


def metrics(y_true, y_pred, y_prob=None):
    out = {
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
        'P': float(precision_score(y_true, y_pred, zero_division=0)),
        'R': float(recall_score(y_true, y_pred, zero_division=0)),
        'Acc': float(accuracy_score(y_true, y_pred)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            out['AUC'] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            out['AUC'] = float('nan')
    return out


def fit_predict(Xtr, ytr, Xte, seed=42, scale=False):
    """返回 (y_pred, y_prob, model)"""
    if scale:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)
    m = make_model(seed=seed)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    try:
        prob = m.predict_proba(Xte)[:, 1]
    except Exception:
        prob = pred.astype(float)
    return pred, prob, m


def cross_llm_eval(train_rows, test_rows, feat_fn, dim=15, seed=42, scale=False):
    """严格三方划分：在 train_rows 上训练，在 test_rows（不同LLM）上测试"""
    Xtr, keys = feat_fn([r['text'] for r in train_rows], dim=dim)
    ytr = np.array([r['label'] for r in train_rows])
    Xte, _ = feat_fn([r['text'] for r in test_rows], dim=dim)
    yte = np.array([r['label'] for r in test_rows])
    pred, prob, m = fit_predict(Xtr, ytr, Xte, seed=seed, scale=scale)
    return metrics(yte, pred, prob), (pred, prob, yte, m)


def kfold_eval(rows, feat_fn, dim=15, k=5, seed=42, scale=False, fold_fn=None):
    """在给定划分下做 K 折评估，返回每折 F1 列表与汇总"""
    from .dataio import pair_stratified_kfold
    fold_fn = fold_fn or pair_stratified_kfold
    X, keys = feat_fn([r['text'] for r in rows], dim=dim)
    y = np.array([r['label'] for r in rows])
    folds = fold_fn(rows, k=k, seed=seed)
    per_fold = []
    for tr, te in folds:
        pred, prob, _ = fit_predict(X[tr], y[tr], X[te], seed=seed, scale=scale)
        per_fold.append(metrics(y[te], pred, prob))
    return per_fold


def summarize(per_fold, key='F1'):
    v = np.array([f[key] for f in per_fold], dtype=float)
    return {'mean': float(v.mean()), 'std': float(v.std(ddof=0)), 'std_sample': float(v.std(ddof=1)),
            'values': [round(float(x), 4) for x in v], 'n': len(v)}
