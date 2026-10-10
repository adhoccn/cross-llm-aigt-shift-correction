# -*- coding: utf-8 -*-
"""
部署侧误报约束下的检测性能：TPR@FPR 与阈值行为。

现有指标只有 macro-F1（隐含阈值 z=0）与 AUC（整体排序），二者都回答不了
部署方最关心的问题：当成批作业交上来，会错怪多少人类作者。本实验在同一批
已算出的分数上补三组指标：
  1. TPR@FPR∈{1%,5%,10%}——固定误报预算下的检出率，检测器领域的通用口径；
  2. FPR@z=0——实际使用的阈值上的误报率，直接对应"错怪多少人类"；
  3. ECE——概率校准误差，衡量 P(AI) 是否可直接作为置信度使用。

校正把整批特征做仿射/截断裂换，分数排序会变，因此 AUC 与 TPR@FPR 都会随之
改变，这正是要测的：校正是把人类侧推回阈值负侧，而不是整体平移。

口径一致性：auto/oracle 的变体名直接取自 E30_tta_select.json，并复算其
5-fold×5 seeds macro-F1 与存档值比对（存档一致性校验），确保新指标与表 11
出自同一次预测。

输出：results/E36_fpr_metrics.json
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from scipy import stats as sps

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
FPR_LEVELS = [0.01, 0.05, 0.10]
N_BOOT = 400
BOOT_SEED = 20261010
OUT_PATH = HERE / 'results' / 'E36_fpr_metrics.json'


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def macro_f1(y, p):
    out = []
    for c in (0, 1):
        tp = float(((p == c) & (y == c)).sum())
        fp = float(((p == c) & (y != c)).sum())
        fn = float(((p != c) & (y == c)).sum())
        denom = 2 * tp + fp + fn
        out.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(out))


def cv_f1(y, p):
    """与 E30 同口径：5-fold×5 seeds 折外 macro-F1 均值。"""
    vals = []
    for seed in SEEDS:
        for _, idx in StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                      random_state=seed).split(np.zeros(len(y)), y):
            vals.append(macro_f1(y[idx], p[idx]))
    return float(np.mean(vals))


def _tpr_at_fpr(z1, z0, alpha):
    """在人类分数上取 (1−α) 分位数作阈值，返回 AI 侧检出率。"""
    if len(z0) == 0 or len(z1) == 0:
        return float('nan')
    thr = float(np.quantile(z0, 1.0 - alpha))
    return float(np.mean(z1 > thr))


def op_metrics(y, z, pb, boot=False, rng=None):
    """一组分数上的完整工作点指标。y∈{0,1}，z 越大越像 AI。"""
    y = np.asarray(y); z = np.asarray(z, dtype=np.float64)
    z0, z1 = z[y == 0], z[y == 1]
    m = {'F1': macro_f1(y, (z > 0).astype(int)),
         'AUC': float(roc_auc_score(y, z)),
         'FPR_z0': float(np.mean(z0 > 0)),
         'TPR_z0': float(np.mean(z1 > 0)),
         'ECE': ece(y, pb)}
    for a in FPR_LEVELS:
        m['TPR@FPR=%g' % a] = _tpr_at_fpr(z1, z0, a)
    if boot and rng is not None:
        n = len(y)
        bs = {('TPR@FPR=%g' % a): [] for a in FPR_LEVELS}
        bs['FPR_z0'] = []
        idx_all = np.arange(n)
        for _ in range(N_BOOT):
            s = rng.choice(idx_all, size=n, replace=True)
            yy, zz = y[s], z[s]
            if len(np.unique(yy)) < 2:
                continue
            a0, a1 = zz[yy == 0], zz[yy == 1]
            for a in FPR_LEVELS:
                bs['TPR@FPR=%g' % a].append(_tpr_at_fpr(a1, a0, a))
            bs['FPR_z0'].append(float(np.mean(a0 > 0)))
        for k, v in bs.items():
            v = np.asarray(v, dtype=np.float64)
            m[k + '_se'] = round(float(v.std(ddof=1)), 4) if len(v) > 2 else None
            m[k + '_lo'] = round(float(np.quantile(v, 0.025)), 4) if len(v) else None
            m[k + '_hi'] = round(float(np.quantile(v, 0.975)), 4) if len(v) else None
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def ece(y, pb, n_bins=10):
    """期望校准误差：置信度分箱后 |平均置信度 − 准确率| 的加权均值。"""
    y = np.asarray(y); pb = np.asarray(pb, dtype=np.float64)
    pb = np.clip(pb, 0.0, 1.0)
    conf = np.maximum(pb, 1.0 - pb)
    pred = (pb >= 0.5).astype(int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    tot = 0.0
    for i in range(n_bins):
        sel = (conf > edges[i]) & (conf <= edges[i + 1])
        if not sel.any():
            continue
        tot += sel.mean() * abs(float(conf[sel].mean()) - float((pred[sel] == y[sel]).mean()))
    return float(tot)


def source_oof_scores(fs_name, cg_h, cg_a, n_tr=N_CG):
    """源域折外分数：在 ChatGPT 上做 5-fold CV，返回 (人类折外分数, AI 折外分数)。

    部署方手上一定有源域留出数据，因此源域人类分数的分位数是先验已知的，
    不需要任何目标域标签。用它作阈值即可尝试在无标签条件下命中误报预算。
    """
    texts = list(cg_h) + list(cg_a)
    y = np.array([0] * len(cg_h) + [1] * len(cg_a))
    z0, z1 = [], []
    for seed in SEEDS[:1]:
        for tr_i, te_i in StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                          random_state=seed).split(np.zeros(len(y)), y):
            f, sc, _, clf, _, _, _, _ = E29.build_pipe(
                fs_name, [texts[i] for i in tr_i], y[tr_i])
            B = np.asarray(f.transform([texts[i] for i in te_i]), dtype=np.float64) \
                if not isinstance(f, E28.CachedRoberta) else f.embed('src_oof', [texts[i] for i in te_i])
            B = sc.transform(np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0))
            z = clf.decision_function(B)
            yy = y[te_i]
            z0.extend(z[yy == 0].tolist()); z1.extend(z[yy == 1].tolist())
    return np.asarray(z0), np.asarray(z1)


def fpr_budget(z0_src, y, z_by_variant, variants):
    """用源域人类分数分位数作阈值，测各变体在目标域上实际命中的 FPR 与 TPR。"""
    out = {}
    y = np.asarray(y)
    for v in variants:
        z = np.asarray(z_by_variant[v], dtype=np.float64)
        row = {}
        for a in FPR_LEVELS:
            tau = float(np.quantile(z0_src, 1.0 - a))
            fpr = float(np.mean(z[y == 0] > tau))
            tpr = float(np.mean(z[y == 1] > tau))
            row['a=%g' % a] = {'tau': round(tau, 4), 'FPR_realized': round(fpr, 4),
                               'TPR': round(tpr, 4), 'budget_err': round(fpr - a, 4)}
        out[v] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='deepseek,qwen,glm,gpt-4o,llama3')
    ap.add_argument('--feats', default='15D,char_2_4')
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--boot', type=int, default=1, help='是否做 Bootstrap（1=做）')
    ap.add_argument('--out', default=str(OUT_PATH))
    args = ap.parse_args()

    targets = args.targets.split(',')
    keeps = [0.3, 0.5, 0.7]
    lams = [0.5, 1.0]
    t0 = time.time()
    log('FPR 约束指标 E36  目标=%s' % targets)

    e30 = json.loads(io.open(HERE / 'results' / 'E30_tta_select.json',
                             encoding='utf-8').read())
    hc3 = E17.build_hc3_index()
    data = {}
    for m in ['chatgpt'] + targets:
        hs, ais, _ = E17.build_pairs(m, hc3)
        data[m] = {'human': hs, 'ai': ais}
        log('  %-10s n=%d' % (m, len(ais)))

    cg = data['chatgpt']
    tr_texts = list(cg['human'][:N_CG]) + list(cg['ai'][:N_CG])
    ytr = np.array([0] * N_CG + [1] * N_CG)

    out = {'protocol': ('ChatGPT-only(%d) 训练；目标批为各目标 LLM 全部人机配对文本；'
                        '分数取校正后 decision_function；macro-F1 与 E30 同为 5-fold×5 seeds'
                        '折外均值；TPR@FPR 在整批上估计（折内样本量不足以支撑 1%% 分位数），'
                        '并用 %d 次 Bootstrap 给标准误与 95%% 区间；ECE 按置信度 10 箱'
                        % (N_CG, N_BOOT)),
           'fpr_levels': FPR_LEVELS, 'n_boot': N_BOOT,
           'feature_sets': {}}

    for fs_name in args.feats.split(','):
        if fs_name not in E28.FEATURES:
            continue
        log('── %s ──' % fs_name)
        f, sc, A, clf, mu_tr, sd_tr, mu0_tr, mu1_tr = E29.build_pipe(fs_name, tr_texts, ytr)
        z0_src, z1_src = source_oof_scores(fs_name, cg['human'], cg['ai'])
        log('  源域折外人类分数 n=%d  99%%/95%%/90%% 分位数=%.3f/%.3f/%.3f'
            % (len(z0_src), *[float(np.quantile(z0_src, 1 - a)) for a in FPR_LEVELS]))
        rng = np.random.default_rng(BOOT_SEED)
        fs_out = {}
        for tgt in targets:
            th, ta = data[tgt]['human'], data[tgt]['ai']
            te_texts = list(th) + list(ta)
            yte = np.array([0] * len(th) + [1] * len(ta))
            B0 = np.asarray(f.transform(te_texts), dtype=np.float64) \
                if not isinstance(f, E28.CachedRoberta) else f.embed('tgt_%s' % tgt, te_texts)
            B0 = sc.transform(np.nan_to_num(B0, nan=0.0, posinf=0.0, neginf=0.0))

            var = {'none': B0, 'bn': E26.tta_bn(B0, mu_tr, sd_tr), 'bnc': E29._bnc(B0, mu_tr)}
            for lam in lams:
                var['em_l%g' % lam] = E29.tta_robust(B0, clf, mu_tr, mu0_tr, mu1_tr,
                                                     args.iters, lam, 'em')
                for k in keeps:
                    var['conf_l%g_k%.2f' % (lam, k)] = E29.tta_robust(
                        B0, clf, mu_tr, mu0_tr, mu1_tr, args.iters, lam, 'conf', k)

            z_ref = clf.decision_function(B0)
            cell = {}
            for v, B in var.items():
                z = clf.decision_function(B)
                pb = clf.predict_proba(B)[:, 1]
                p = (z > 0).astype(int)
                mm = op_metrics(yte, z, pb, boot=bool(args.boot), rng=rng)
                mm['F1_cv'] = round(cv_f1(yte, p), 4)   # 与 E30 存档口径对齐
                # 排序是否改变：校正若是分数空间的单调重标定，则 AUC 与 TPR@FPR
                # 严格不变，收益全部来自阈值位置——这是本实验最关键的判据
                rho = float(sps.spearmanr(z_ref, z).statistic)
                mm['spearman_vs_none'] = round(rho, 6)
                mm['auc_delta'] = round(mm['AUC'] - cell['none']['AUC'], 6) \
                    if 'none' in cell else 0.0
                cell[v] = mm

            ch = e30['feature_sets'][fs_name]['alpha_0'].get(tgt + '_choice', {})
            v_auto = ch.get('auto')
            v_orac = ch.get('oracle')
            # 存档一致性校验：复算的 5-fold×5seeds F1 应与 E30 存档一致
            chk = {}
            for nm, v in [('auto', v_auto), ('oracle', v_orac), ('none', 'none')]:
                if v not in cell:
                    continue
                ref = {'auto': ch.get('auto_F1'), 'oracle': ch.get('oracle_F1'),
                       'none': ch.get('none_F1')}[nm]
                chk[nm] = {'variant': v, 'recomputed': cell[v]['F1_cv'],
                           'archived': ref,
                           'abs_diff': None if ref is None
                                       else round(abs(cell[v]['F1_cv'] - ref), 4)}

            # 无标签误报预算：阈值取源域人类分数的 (1−α) 分位数，看实际命中多少
            fb = fpr_budget(z0_src, yte, {v: clf.decision_function(B)
                                          for v, B in var.items()},
                            [x for x in ['none', v_orac, v_auto] if x in var])
            fs_out[tgt] = {'n': int(len(yte)), 'n_human': int((yte == 0).sum()),
                           'cells': cell, 'chosen': {'auto': v_auto, 'oracle': v_orac},
                           'archive_check': chk, 'fpr_budget': fb}
            g = cell[v_orac] if v_orac in cell else cell['none']
            n0 = cell['none']
            log('   %-9s n=%d  none: F1=%.3f AUC=%.4f FPR@z0=%.3f TPR@1%%=%.3f  |  '
                'oracle(%s): F1=%.3f AUC=%.4f FPR@z0=%.3f TPR@1%%=%.3f  ρ=%.4f'
                % (tgt, len(yte), n0['F1'], n0['AUC'], n0['FPR_z0'], n0['TPR@FPR=0.01'],
                   v_orac, g['F1'], g['AUC'], g['FPR_z0'], g['TPR@FPR=0.01'],
                   g['spearman_vs_none']))
            bad = [k for k, v in chk.items()
                   if v['abs_diff'] is not None and v['abs_diff'] > 0.002]
            if bad:
                log('   ⚠ 存档不一致：%s' % bad)
        out['feature_sets'][fs_name] = fs_out

    out['runtime_sec'] = round(time.time() - t0, 1)
    OUT = Path(args.out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    io.open(OUT, 'w', encoding='utf-8').write(json.dumps(out, ensure_ascii=False, indent=1))
    log('写出 %s  (%.1fs)' % (OUT, out['runtime_sec']))


if __name__ == '__main__':
    main()
