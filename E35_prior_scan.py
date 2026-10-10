# -*- coding: utf-8 -*-
"""
类先验失衡扫描：UBC 在多失衡时开始失效，以及失效是否可检测。

UBC 隐含假设批次类先验接近 1:1，失衡时类平衡先验会把候选判为
无效、系统退回不校正，且该失效"可检测而非静默"。本实验给出该性质的定量依据。
本实验按 AI 占比 π 构造失衡批次，跑完整 UBC 流程，测量：
  1. none / auto（GMM + 类平衡先验）/ oracle 三者的 macro-F1 随 π 的变化；
  2. auto 相对 none 是否仍然有益（收益转负即失效）；
  3. 类平衡告警（|ŵ−0.5| > 0.08）的触发时机是否早于性能塌陷；
  4. 系统是否如设计那样退回不校正。

输出：results/E35_prior_scan.json
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
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold, KFold

_spec = importlib.util.spec_from_file_location('e26', HERE / 'E26_tta.py')
E26 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(E26)
_spec2 = importlib.util.spec_from_file_location('e17', HERE / 'E17_crossllm_ghostbuster.py')
E17 = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(E17)
_spec3 = importlib.util.spec_from_file_location('e28', HERE / 'E28_tta_lambda.py')
E28 = importlib.util.module_from_spec(_spec3); _spec3.loader.exec_module(E28)
_spec4 = importlib.util.spec_from_file_location('e29', HERE / 'E29_tta_robust.py')
E29 = importlib.util.module_from_spec(_spec4); _spec4.loader.exec_module(E29)
_spec5 = importlib.util.spec_from_file_location('e30', HERE / 'E30_tta_select.py')
E30 = importlib.util.module_from_spec(_spec5); _spec5.loader.exec_module(E30)

SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5
N_CG = 800
BAL_TOL = 0.08          # 类平衡先验阈值（对应论文 §5.4）
OUT_PATH = HERE / 'results' / 'E35_prior_scan.json'


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def macro_f1(y, p):
    """与 sklearn average='macro', labels=[0,1] 同口径：缺失类的 F1 记 0。"""
    out = []
    for c in (0, 1):
        tp = float(((p == c) & (y == c)).sum())
        fp = float(((p == c) & (y != c)).sum())
        fn = float(((p != c) & (y == c)).sum())
        denom = 2 * tp + fp + fn
        out.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(out))


def cv_f1(yte, pred):
    """5-fold×5 seeds 的折外 macro-F1 均值（单类时退化为不分层 KFold）。"""
    vals = []
    for seed in SEEDS:
        if len(np.unique(yte)) < 2:
            folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(yte)))
        else:
            folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                    random_state=seed).split(np.zeros(len(yte)), yte)
        for _, idx in folds:
            vals.append(macro_f1(yte[idx], pred[idx]))
    return float(np.mean(vals))


def cv_recall_ai(yte, pred):
    """AI 类召回的折外均值。

    π→1 时批次里没有人类样本，macro-F1 退化为 AI 类 F1 的一半、判 0 的那类恒为
    0，横向不可比；此时唯一有语义的指标是"AI 文本被判为 AI 的比例"。
    """
    vals = []
    for seed in SEEDS:
        if len(np.unique(yte)) < 2:
            folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(yte)))
        else:
            folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                    random_state=seed).split(np.zeros(len(yte)), yte)
        for _, idx in folds:
            y, p = yte[idx], pred[idx]
            tp = float(((p == 1) & (y == 1)).sum())
            fn = float(((p != 1) & (y == 1)).sum())
            vals.append(tp / max(tp + fn, 1e-9))
    return float(np.mean(vals))


def cv_pred_ai_rate(pred):
    """预测为 AI 的比例：与真实 π 的偏离是类平衡先验的直接观测量。"""
    return float(np.mean(np.asarray(pred) == 1))


def scan_once(hs, ais, pi, n_total, seed, f, sc, clf, mu_tr, sd_tr, mu0_tr, mu1_tr,
              iters, keeps, lams):
    """抽一个 π 批次并跑完整 UBC，返回该次重复的完整观测。"""
    rng = np.random.default_rng(seed)
    te_texts, yte = build_batch(hs, ais, pi, n_total, rng)
    B0 = np.asarray(f.transform(te_texts), dtype=np.float64) \
        if not isinstance(f, E28.CachedRoberta) else f.embed('s%d_%g' % (seed, pi), te_texts)
    B0 = sc.transform(np.nan_to_num(B0, nan=0.0, posinf=0.0, neginf=0.0))
    var = variants_for(B0, clf, mu_tr, sd_tr, mu0_tr, mu1_tr, iters, keeps, lams)
    cell = {}
    for v, B in var.items():
        z = clf.decision_function(B)
        p = clf.predict(B)
        pb = clf.predict_proba(B)[:, 1]
        f_gmm, _ = E30.pred_gmm(z)
        cell[v] = {'F1_obs': round(cv_f1(yte, p), 4),
                   'recall_ai': round(cv_recall_ai(yte, p), 4),
                   'pred_ai_rate': round(cv_pred_ai_rate(p), 4),
                   'F1_pred_gmm': round(f_gmm, 4),
                   'imbalance': round(float(abs(pb.mean() - 0.5)), 4)}
    def score_gmm(v):
        return cell[v]['F1_pred_gmm'] if cell[v]['imbalance'] < BAL_TOL else -1.0
    n_valid = sum(1 for v in cell if score_gmm(v) > -1.0)
    best_auto = max(cell, key=score_gmm)
    best_orac = max(cell, key=lambda v: cell[v]['F1_obs'])
    return {
        'F1_none': cell['none']['F1_obs'],
        'F1_auto': cell[best_auto]['F1_obs'],
        'F1_oracle': cell[best_orac]['F1_obs'],
        'recall_none': cell['none']['recall_ai'],
        'recall_auto': cell[best_auto]['recall_ai'],
        'pred_ai_rate_auto': cell[best_auto]['pred_ai_rate'],
        'imbalance_none': cell['none']['imbalance'],
        'imbalance_auto': cell[best_auto]['imbalance'],
        'auto': best_auto, 'oracle': best_orac,
        'n_valid_candidates': n_valid,
        'revert_to_none': (best_auto == 'none'),
        'gain': round(cell[best_auto]['F1_obs'] - cell['none']['F1_obs'], 4),
        'hit_oracle': best_auto == best_orac}


def build_batch(hs, ais, pi, n_total, rng):
    """按 AI 占比 π 抽样出一个失衡批次。"""
    n_ai = min(int(round(n_total * pi)), len(ais))
    n_hu = min(n_total - int(round(n_total * pi)), len(hs))
    ai_i = rng.choice(len(ais), size=n_ai, replace=n_ai > len(ais))
    hu_i = rng.choice(len(hs), size=n_hu, replace=n_hu > len(hs))
    texts = [ais[i] for i in ai_i] + [hs[i] for i in hu_i]
    y = np.array([1] * n_ai + [0] * n_hu)
    return texts, y


def variants_for(B0, clf, mu_tr, sd_tr, mu0_tr, mu1_tr, iters, keeps, lams):
    var = {'none': B0,
           'bn': E26.tta_bn(B0, mu_tr, sd_tr),
           'bnc': E29._bnc(B0, mu_tr)}
    for lam in lams:
        var['em_l%g' % lam] = E29.tta_robust(B0, clf, mu_tr, mu0_tr, mu1_tr, iters, lam, 'em')
        for k in keeps:
            var['conf_l%g_k%.2f' % (lam, k)] = E29.tta_robust(
                B0, clf, mu_tr, mu0_tr, mu1_tr, iters, lam, 'conf', k)
    return var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='deepseek,glm')
    ap.add_argument('--feats', default='15D,char_2_4')
    ap.add_argument('--pis', default='0.5,0.6,0.7,0.8,0.9,1.0')
    ap.add_argument('--batch', type=int, default=600)
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--reps', type=int, default=5,
                    help='每个 π 的独立抽样重复次数（单次抽样噪声大，必须重复）')
    ap.add_argument('--out', default=str(OUT_PATH))
    args = ap.parse_args()

    targets = args.targets.split(',')
    pis = [float(x) for x in args.pis.split(',')]
    keeps = [0.3, 0.5, 0.7]
    lams = [0.5, 1.0]

    t0 = time.time()
    log('类先验失衡扫描 E35  π=%s  批次=%d  iters=%d  reps=%d'
        % (pis, args.batch, args.iters, args.reps))
    hc3 = E17.build_hc3_index()
    data = {}
    for m in ['chatgpt'] + targets:
        hs, ais, _ = E17.build_pairs(m, hc3)
        data[m] = {'human': hs, 'ai': ais}
        log('  %-10s n=%d' % (m, len(ais)))

    cg = data['chatgpt']
    tr_texts = list(cg['human'][:N_CG]) + list(cg['ai'][:N_CG])
    ytr = np.array([0] * N_CG + [1] * N_CG)

    out = {'protocol': ('ChatGPT-only(%d) 训练；目标批按 AI 占比 π 抽样（总数 %d，'
                        '每格 %d 次独立抽样）；11 个校正变体；'
                        'auto=GMM 预测 F1 + 类平衡先验(|ŵ−0.5|>%.2f 判无效)；'
                        'macro-F1 与 AI 召回均由 5-fold×5 seeds 折外估计'
                        % (N_CG, args.batch, args.reps, BAL_TOL)),
           'pis': pis, 'batch_size': args.batch, 'reps': args.reps,
           'imbalance_tol': BAL_TOL, 'feature_sets': {}}

    for fs_name in args.feats.split(','):
        if fs_name not in E28.FEATURES:
            continue
        log('── %s ──' % fs_name)
        f, sc, A, clf, mu_tr, sd_tr, mu0_tr, mu1_tr = E29.build_pipe(fs_name, tr_texts, ytr)
        fs_out = {}
        for tgt in targets:
            hs, ais = data[tgt]['human'], data[tgt]['ai']
            tgt_out = {}
            for pi in pis:
                reps = []
                for rep in range(args.reps):
                    seed = 1000 + int(pi * 100) + rep * 17
                    reps.append(scan_once(hs, ais, pi, args.batch, seed, f, sc, clf,
                                          mu_tr, sd_tr, mu0_tr, mu1_tr,
                                          args.iters, keeps, lams))

                def agg(key):
                    vals = [r[key] for r in reps]
                    return (round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4))

                n_m, n_s = agg('F1_none')
                a_m, a_s = agg('F1_auto')
                o_m, o_s = agg('F1_oracle')
                gn_m, gn_s = agg('gain')
                rc_n, _ = agg('recall_none')
                rc_a, _ = agg('recall_auto')
                imb, _ = agg('imbalance_none')
                nval, _ = agg('n_valid_candidates')
                revert_rate = round(float(np.mean([r['revert_to_none'] for r in reps])), 4)
                worse_rate = round(float(np.mean([r['gain'] < -1e-9 for r in reps])), 4)
                hit_rate = round(float(np.mean([r['hit_oracle'] for r in reps])), 4)
                tgt_out['pi_%g' % pi] = {
                    'reps': reps,
                    'F1_none': n_m, 'F1_none_std': n_s,
                    'F1_auto': a_m, 'F1_auto_std': a_s,
                    'F1_oracle': o_m, 'F1_oracle_std': o_s,
                    'gain': gn_m, 'gain_std': gn_s,
                    'recall_none': rc_n, 'recall_auto': rc_a,
                    'imbalance_none': imb, 'n_valid_candidates': nval,
                    'revert_rate': revert_rate, 'worse_rate': worse_rate,
                    'hit_oracle_rate': hit_rate}
                log('   %-9s π=%.1f  none=%.4f±%.4f  auto=%.4f±%.4f  oracle=%.4f  '
                    '增益=%+.4f±%.4f  AI召回 %.3f→%.3f  失衡=%.3f  候选=%.1f  '
                    '退回%.0f%% 劣化%.0f%% 命中%.0f%%'
                    % (tgt, pi, n_m, n_s, a_m, a_s, o_m, gn_m, gn_s, rc_n, rc_a,
                       imb, nval, 100 * revert_rate, 100 * worse_rate, 100 * hit_rate))
            fs_out[tgt] = tgt_out
        out['feature_sets'][fs_name] = fs_out

    out['runtime_sec'] = round(time.time() - t0, 1)
    OUT = Path(args.out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    io.open(OUT, 'w', encoding='utf-8').write(json.dumps(out, ensure_ascii=False, indent=1))
    log('写出 %s  (%.1fs)' % (OUT, out['runtime_sec']))


if __name__ == '__main__':
    main()
