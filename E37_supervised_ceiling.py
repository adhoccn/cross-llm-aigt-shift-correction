# -*- coding: utf-8 -*-
"""
目标域监督上界与恢复率：零标签校正到底换回了多少本要标注才能拿回的损失。

"F1 从 0.793 恢复到 0.900"这个说法缺少一个参照——0.900 距离"给这个目标 LLM
打上标签重新训练"能到多少还有多远。本实验补上这个上界，并把总损失分解为
两段：
    总损失   F1_sup  − F1_none      （跨域造成的全部损失）
    可恢复段 F1_TTA  − F1_none      （无需任何目标标签即可拿回）
    残余段   F1_sup  − F1_TTA       （只有拿到目标标签才能拿回）
    恢复率   (F1_TTA − F1_none) / (F1_sup − F1_none)

E36 已证明类条件对齐是分数空间的严格单调重标定（ρ=1，ΔAUC=0），因此可恢复段
应当恰好等同于"阈值位置"，残余段等同于"排序质量"。本实验同时报告三者的 AUC，
用 AUC 恢复率是否为 0 来独立验证这一分解。

协议：目标域内做 5-fold×5 seeds 交叉验证，每折以 800+800 条目标域样本训练、
在留出折上测试；无校正臂用同样规模的 ChatGPT-only 训练集、在同一批留出折上
测试，保证训练规模与测试集完全一致。TTA 臂在留出折（400 条）上无标签执行
类条件对齐并用该折真值挑最优变体（折内 oracle）。

输出：results/E37_supervised_ceiling.json
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

_spec = importlib.util.spec_from_file_location('e17', HERE / 'E17_crossllm_ghostbuster.py')
E17 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(E17)
_spec2 = importlib.util.spec_from_file_location('e26', HERE / 'E26_tta.py')
E26 = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(E26)
_spec3 = importlib.util.spec_from_file_location('e28', HERE / 'E28_tta_lambda.py')
E28 = importlib.util.module_from_spec(_spec3); _spec3.loader.exec_module(E28)
_spec4 = importlib.util.spec_from_file_location('e29', HERE / 'E29_tta_robust.py')
E29 = importlib.util.module_from_spec(_spec4); _spec4.loader.exec_module(E29)
_spec5 = importlib.util.spec_from_file_location('e30', HERE / 'E30_tta_select.py')
E30 = importlib.util.module_from_spec(_spec5); _spec5.loader.exec_module(E30)

SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5
N_TR_PER_CLASS = 800        # 与 E30 的 N_CG 一致，保证训练规模可比
OUT_PATH = HERE / 'results' / 'E37_supervised_ceiling.json'


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


def fit_clf(fs_name, tr_texts, ytr, cache_tag=None):
    """在给定训练集上拟合 (特征, 标准化, 逻辑回归)。TF-IDF 只在训练集内拟合。"""
    f = E28.FEATURES[fs_name]()
    if isinstance(f, E28.CachedRoberta):
        A = f.embed(cache_tag or 'tr', tr_texts)
    else:
        A = np.asarray(f.fit(tr_texts).transform(tr_texts), dtype=np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    sc = StandardScaler().fit(A)
    A = sc.transform(A)
    clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0).fit(A, ytr)
    return f, sc, clf, A


def transform(f, sc, texts, cache_tag=None):
    B = np.asarray(f.transform(texts), dtype=np.float64) \
        if not isinstance(f, E28.CachedRoberta) else f.embed(cache_tag, texts)
    return sc.transform(np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='deepseek,qwen,glm,gpt-4o,llama3')
    ap.add_argument('--feats', default='15D,char_2_4')
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--out', default=str(OUT_PATH))
    args = ap.parse_args()

    targets = args.targets.split(',')
    keeps = [0.3, 0.5, 0.7]
    lams = [0.5, 1.0]
    t0 = time.time()
    log('目标域监督上界 E37  目标=%s' % targets)

    e30 = json.loads(io.open(HERE / 'results' / 'E30_tta_select.json',
                             encoding='utf-8').read())
    hc3 = E17.build_hc3_index()
    data = {}
    for m in ['chatgpt'] + targets:
        hs, ais, _ = E17.build_pairs(m, hc3)
        data[m] = {'human': hs, 'ai': ais}
        log('  %-10s n=%d' % (m, len(ais)))

    cg = data['chatgpt']
    cg_texts = list(cg['human'][:N_TR_PER_CLASS]) + list(cg['ai'][:N_TR_PER_CLASS])
    y_cg = np.array([0] * N_TR_PER_CLASS + [1] * N_TR_PER_CLASS)

    out = {'protocol': ('目标域内 5-fold×5 seeds 交叉验证：每折以 %d+%d 条目标域样本'
                        '训练、留出折测试；无校正臂用同规模 ChatGPT-only(%d+%d) 训练集'
                        '在同一批留出折上测试；TTA 臂在留出折上无标签执行类条件对齐，'
                        '并用该折真值挑最优变体（折内 oracle）。TF-IDF 词表与标准化'
                        '参数只在训练折内拟合。F1 为 macro-F1。'
                        % (N_TR_PER_CLASS, N_TR_PER_CLASS,
                           N_TR_PER_CLASS, N_TR_PER_CLASS)),
           'n_train_per_class': N_TR_PER_CLASS, 'feature_sets': {}}

    for fs_name in args.feats.split(','):
        if fs_name not in E28.FEATURES:
            continue
        log('── %s ──' % fs_name)
        fs_out = {}
        for tgt in targets:
            th, ta = data[tgt]['human'], data[tgt]['ai']
            te_texts = list(th) + list(ta)
            yte = np.array([0] * len(th) + [1] * len(ta))

            f_cg, sc_cg, clf_cg, A_cg = fit_clf(fs_name, cg_texts, y_cg, 'train_shared')
            mu_tr, sd_tr = A_cg.mean(axis=0), A_cg.std(axis=0)
            mu0_tr, mu1_tr = A_cg[y_cg == 0].mean(axis=0), A_cg[y_cg == 1].mean(axis=0)

            rec = {'none': [], 'sup': [], 'tta': [], 'tta_auto': [],
                   'auc_none': [], 'auc_sup': [], 'auc_tta': [], 'auc_tta_auto': [],
                   'tta_variant': [], 'tta_variant_auto': []}
            for seed in SEEDS:
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                for k, (tr_i, te_i) in enumerate(skf.split(np.zeros(len(yte)), yte)):
                    y_te = yte[te_i]
                    # ① 无校正：ChatGPT-only 分类器
                    B_te = transform(f_cg, sc_cg, [te_texts[i] for i in te_i],
                                     'tgt_%s' % tgt)
                    z = clf_cg.decision_function(B_te)
                    rec['none'].append(macro_f1(y_te, (z > 0).astype(int)))
                    rec['auc_none'].append(float(roc_auc_score(y_te, z)))

                    # ② TTA：在该折上无标签类条件对齐，折内 oracle 选变体
                    #    变体集与 E30 完全一致（含 none/bn），保证 oracle 口径可比
                    var = {'none': B_te, 'bn': E26.tta_bn(B_te, mu_tr, sd_tr),
                           'bnc': E29._bnc(B_te, mu_tr)}
                    for lam in lams:
                        var['em_l%g' % lam] = E29.tta_robust(
                            B_te, clf_cg, mu_tr, mu0_tr, mu1_tr, args.iters, lam, 'em')
                        for kk in keeps:
                            var['conf_l%g_k%.2f' % (lam, kk)] = E29.tta_robust(
                                B_te, clf_cg, mu_tr, mu0_tr, mu1_tr, args.iters, lam, 'conf', kk)
                    # oracle 臂用该折真值挑变体（上界，但选参本身已用到标签），
                    # auto 臂用一维 GMM 无标签挑变体——后者才是真正的零标签结果
                    best, best_v = -1.0, 'none'
                    best_g, best_va = -1.0, 'none'
                    best_z = best_za = None
                    for v, B in var.items():
                        zz = clf_cg.decision_function(B)
                        s = macro_f1(y_te, (zz > 0).astype(int))
                        if s > best:
                            best, best_v, best_z = s, v, zz
                        fg, _ = E30.pred_gmm(zz)
                        if fg == fg and fg > best_g:
                            best_g, best_va, best_za = fg, v, zz
                    rec['tta'].append(best); rec['tta_variant'].append(best_v)
                    rec['auc_tta'].append(float(roc_auc_score(y_te, best_z)))
                    rec['tta_auto'].append(macro_f1(y_te, (best_za > 0).astype(int)))
                    rec['tta_variant_auto'].append(best_va)
                    rec['auc_tta_auto'].append(float(roc_auc_score(y_te, best_za)))

                    # ③ 监督上界：同折内用目标域标签重新训练
                    tr_sub = [te_texts[i] for i in tr_i]
                    y_tr = yte[tr_i]
                    # 训练规模对齐：每类取 N_TR_PER_CLASS（不足则全取）
                    idx0 = np.where(y_tr == 0)[0][:N_TR_PER_CLASS]
                    idx1 = np.where(y_tr == 1)[0][:N_TR_PER_CLASS]
                    sel = np.concatenate([idx0, idx1])
                    f_s, sc_s, clf_s, _ = fit_clf(
                        fs_name, [tr_sub[i] for i in sel], y_tr[sel],
                        'sup_%s_s%d_k%d' % (tgt, seed, k))
                    B_s = transform(f_s, sc_s, [te_texts[i] for i in te_i],
                                    'supte_%s_s%d_k%d' % (tgt, seed, k))
                    zs = clf_s.decision_function(B_s)
                    rec['sup'].append(macro_f1(y_te, (zs > 0).astype(int)))
                    rec['auc_sup'].append(float(roc_auc_score(y_te, zs)))

            def mn(k):
                return float(np.mean(rec[k]))

            def sd(k):
                return float(np.std(rec[k], ddof=1))

            f_none, f_tta, f_sup = mn('none'), mn('tta'), mn('sup')
            f_auto = mn('tta_auto')
            gap = f_sup - f_none
            # 缺口过小（源域训练已接近目标域监督上界）时恢复率无意义，记为不适用
            degenerate = abs(gap) < 0.01
            rec_rate = (f_tta - f_none) / gap if not degenerate else float('nan')
            rec_rate_auto = (f_auto - f_none) / gap if not degenerate else float('nan')
            a_none, a_tta, a_sup = mn('auc_none'), mn('auc_tta'), mn('auc_sup')
            a_auto = mn('auc_tta_auto')
            agap = a_sup - a_none
            rec_auc = (a_tta - a_none) / agap if abs(agap) > 1e-9 else float('nan')

            ch = e30['feature_sets'][fs_name]['alpha_0'].get(tgt + '_choice', {})
            fs_out[tgt] = {
                'n_folds': len(rec['none']),
                'F1_none': round(f_none, 4), 'F1_none_std': round(sd('none'), 4),
                'F1_tta': round(f_tta, 4), 'F1_tta_std': round(sd('tta'), 4),
                'F1_tta_auto': round(f_auto, 4), 'F1_tta_auto_std': round(sd('tta_auto'), 4),
                'F1_sup': round(f_sup, 4), 'F1_sup_std': round(sd('sup'), 4),
                'gap_total': round(gap, 4),
                'gap_recoverable': round(f_tta - f_none, 4),
                'gap_recoverable_auto': round(f_auto - f_none, 4),
                'gap_residual': round(f_sup - f_tta, 4),
                'gap_residual_auto': round(f_sup - f_auto, 4),
                'recovery_rate': round(rec_rate, 4) if rec_rate == rec_rate else None,
                'recovery_rate_auto': round(rec_rate_auto, 4)
                                    if rec_rate_auto == rec_rate_auto else None,
                'degenerate_gap': bool(degenerate),
                'AUC_none': round(a_none, 4), 'AUC_tta': round(a_tta, 4),
                'AUC_tta_auto': round(a_auto, 4),
                'AUC_sup': round(a_sup, 4),
                'AUC_gap_total': round(agap, 4),
                'recovery_rate_auc': round(rec_auc, 4) if rec_auc == rec_auc else None,
                'tta_variants': rec['tta_variant'],
                'tta_variants_auto': rec['tta_variant_auto'],
                # 与表 11（整批 n=2000）的交叉校验：协议不同但应接近
                'crosscheck_E30_fullbatch': {
                    'F1_none': ch.get('none_F1'), 'F1_oracle': ch.get('oracle_F1'),
                    'F1_auto': ch.get('auto_F1'),
                    'diff_none': None if ch.get('none_F1') is None
                                 else round(f_none - ch['none_F1'], 4),
                    'diff_tta': None if ch.get('oracle_F1') is None
                                else round(f_tta - ch['oracle_F1'], 4)}}
            log('   %-9s none=%.4f  auto=%.4f  oracle=%.4f  sup=%.4f  |  总缺口=%+.4f  '
                '恢复率 auto=%s oracle=%s  (AUC %.4f→%.4f→%.4f)'
                % (tgt, f_none, f_auto, f_tta, f_sup, gap,
                   '—' if rec_rate_auto != rec_rate_auto else '%.3f' % rec_rate_auto,
                   '—' if rec_rate != rec_rate else '%.3f' % rec_rate,
                   a_none, a_auto, a_sup))
        out['feature_sets'][fs_name] = fs_out

    out['runtime_sec'] = round(time.time() - t0, 1)
    OUT = Path(args.out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    io.open(OUT, 'w', encoding='utf-8').write(json.dumps(out, ensure_ascii=False, indent=1))
    log('写出 %s  (%.1fs)' % (OUT, out['runtime_sec']))


if __name__ == '__main__':
    main()
