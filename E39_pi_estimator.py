# -*- coding: utf-8 -*-
"""
批次类先验的可观测量：$\\hat\\pi$ 能否作为 UBC 的前置判据。

背景：无标签批校正隐含假设批次类先验接近 1:1。§6.5 情形二给出该假设失效的定量边界
（AI 占比 $\\pi\\to1$ 时保护不再触发、校正反而造成损失），因而部署方需要在校正之前
先判断 $\\pi$。本实验回答"用什么判断、判断得有多准"：

  * 观测量为检测器在**未校正**批次上的输出——平均预测概率 $\\bar p$ 与判为 AI 的
    硬判比例 $r_{\\mathrm{AI}}$，二者都无需目标域标签；
  * 测量它们相对真值 $\\pi$ 的偏差方向与幅度（是否存在向 0.5 的系统性收缩）；
  * 把"由 $\\hat\\pi$ 超阈值即拒绝校正"做成一条可操作的拒绝规则，按品位检查它在
    危险区（$\\pi\\ge0.9$）与可用区（$\\pi\\le0.6$）上的触发率。

输出：results/E39_pi_estimator.json
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
from scipy.stats import spearmanr, linregress
from scipy.stats import norm as _norm

_spec2 = importlib.util.spec_from_file_location('e17', HERE / 'E17_crossllm_ghostbuster.py')
E17 = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(E17)
_spec3 = importlib.util.spec_from_file_location('e28', HERE / 'E28_tta_lambda.py')
E28 = importlib.util.module_from_spec(_spec3); _spec3.loader.exec_module(E28)
_spec4 = importlib.util.spec_from_file_location('e29', HERE / 'E29_tta_robust.py')
E29 = importlib.util.module_from_spec(_spec4); _spec4.loader.exec_module(E29)

N_CG = 800                      # ChatGPT-only 训练规模，与主实验一致
DEFAULT_TARGETS = 'deepseek,glm,qwen,gpt-4o,llama3'
DEFAULT_PIS = '0.5,0.6,0.7,0.8,0.9,1.0'
OUT_PATH = HERE / 'results' / 'E39_pi_estimator.json'


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def build_batch(hs, ais, pi, n_total, rng):
    """按 AI 占比 π 抽样出一个失衡批次（与 E35 同规则，便于对照）。"""
    n_ai = min(int(round(n_total * pi)), len(ais))
    n_hu = min(n_total - int(round(n_total * pi)), len(hs))
    ai_i = rng.choice(len(ais), size=n_ai, replace=n_ai > len(ais))
    hu_i = rng.choice(len(hs), size=n_hu, replace=n_hu > len(hs))
    texts = [ais[i] for i in ai_i] + [hs[i] for i in hu_i]
    y = np.array([1] * n_ai + [0] * n_hu)
    return texts, y


def gmm_ai_weight(z):
    """一维两分量 GMM：返回 UBC 内部已需要的五个量。

    返回 (w1, e0, e1)：w1 为高均值分量权重，e0/e1 为该拟合下的两类错误率
    （与 E30.pred_gmm 同一配置）。
    """
    from sklearn.mixture import GaussianMixture
    zz = np.asarray(z, dtype=np.float64).reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=0, max_iter=300,
                         means_init=np.array([[zz.min()], [zz.max()]])).fit(zz)
    order = np.argsort(gm.means_.ravel())
    mu0, mu1 = gm.means_.ravel()[order]
    s0 = np.sqrt(max(gm.covariances_.ravel()[order[0]], 1e-9))
    s1 = np.sqrt(max(gm.covariances_.ravel()[order[1]], 1e-9))
    w1 = float(gm.weights_.ravel()[order[1]])
    e1 = float(_norm.cdf(-mu1 / s1))     # 高分量判为人类
    e0 = float(_norm.cdf(mu0 / s0))      # 低分量判为 AI
    return w1, e0, e1


def measure_once(hs, ais, pi, n_total, seed, f, sc, clf):
    """返回未校正批次上三类无需目标域标签的 π 观测量。"""
    rng = np.random.default_rng(seed)
    texts, y = build_batch(hs, ais, pi, n_total, rng)
    B = f.transform(texts)
    B = sc.transform(np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0))
    proba = clf.predict_proba(B)[:, 1]
    pred = clf.predict(B)
    hard = float(np.mean(pred == 1))
    w1, e0, e1 = gmm_ai_weight(clf.decision_function(B))
    # r_AI = FPR + pi*(TPR-FPR) 在固定阈值下对 pi 仿射；用 GMM 得到的两类错误率反解 pi
    slope = (1.0 - e1) - e0
    inv = np.nan if abs(slope) < 1e-6 else (hard - e0) / slope
    return {'pi_hat_soft': round(float(proba.mean()), 4),   # 平均预测概率 \bar p
            'pi_hat_hard': round(hard, 4),                  # 硬判 AI 比例
            'pi_hat_gmm': round(w1, 4),                     # GMM 权重
            'pi_hat_inv': round(float(np.clip(inv, 0.0, 1.0)), 4),   # 仿射反解
            'pi_true': round(float(np.mean(y == 1)), 4)}


def reject_curve(cells, key, cuts):
    """给定观测量与阈值集合，统计拒绝规则在危险区与可用区的触发率。

    危险区 = 真值 π ≥ 0.9（§6.5 情形二实测增益转负区间）；
    可用区 = 真值 π ≤ 0.6（四组配置增益全为正）。
    """
    out = []
    for c in cuts:
        hit = miss = fp = tn = 0.0
        n_danger = n_usable = 0
        pi_keys = [k for k in cells if k.startswith('pi_')]
        for pi_raw in sorted(pi_keys, key=lambda k: float(k.split('_')[1])):
            pi = float(pi_raw.split('_')[1])
            rec = cells[pi_raw]
            vals = rec['reps_' + key]
            fires = [v > c for v in vals]
            rate = float(np.mean(fires)) if fires else 0.0
            if pi >= 0.9:
                n_danger += 1
                hit += rate
            elif pi <= 0.6:
                n_usable += 1
                fp += rate
        out.append({'cut': c,
                    'fire_danger': round(hit / n_danger, 4) if n_danger else None,
                    'fire_usable': round(fp / n_usable, 4) if n_usable else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default=DEFAULT_TARGETS)
    ap.add_argument('--feats', default='15D,char_2_4')
    ap.add_argument('--pis', default=DEFAULT_PIS)
    ap.add_argument('--batch', type=int, default=600)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--out', default=str(OUT_PATH))
    args = ap.parse_args()

    targets = args.targets.split(',')
    pis = [float(x) for x in args.pis.split(',')]
    t0 = time.time()
    log('类先验观测量 E39  π=%s  批次=%d  reps=%d' % (pis, args.batch, args.reps))

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
                        '每格 %d 次独立抽样）；观测量取**未校正**批次上的平均预测概率 '
                        '$\\bar p$ 与硬判 AI 比例，均不使用目标域标签'
                        % (N_CG, args.batch, args.reps)),
           'pis': pis, 'batch_size': args.batch, 'reps': args.reps,
           'feature_sets': {}}

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
                    seed = 1000 + int(pi * 100) + rep * 17      # 与 E35 同一种子规则
                    reps.append(measure_once(hs, ais, pi, args.batch, seed, f, sc, clf))
                node = {'reps': reps}
                for key in ('pi_hat_soft', 'pi_hat_hard', 'pi_hat_gmm', 'pi_hat_inv'):
                    node['reps_' + key] = [r[key] for r in reps]
                for key in ('pi_hat_soft', 'pi_hat_hard', 'pi_hat_gmm', 'pi_hat_inv'):
                    vals = [r[key] for r in reps]
                    node[key] = round(float(np.mean(vals)), 4)
                    node[key + '_std'] = round(float(np.std(vals)), 4)
                node['pi_true'] = round(float(np.mean([r['pi_true'] for r in reps])), 4)
                tgt_out['pi_%g' % pi] = node
                log('   %-9s π=%.1f  \\bar p=%.4f±%.4f r=%.4f±%.4f w=%.4f±%.4f inv=%.4f±%.4f'
                    % (tgt, pi, node['pi_hat_soft'], node['pi_hat_soft_std'],
                       node['pi_hat_hard'], node['pi_hat_hard_std'],
                       node['pi_hat_gmm'], node['pi_hat_gmm_std'],
                       node['pi_hat_inv'], node['pi_hat_inv_std']))
            # 观测量对真值的判别力（跨全部 π 的格子均值）
            truth = [tgt_out['pi_%g' % p]['pi_true'] for p in pis]
            for key in ('pi_hat_soft', 'pi_hat_hard', 'pi_hat_gmm', 'pi_hat_inv'):
                est = [tgt_out['pi_%g' % p][key] for p in pis]
                rho = spearmanr(truth, est).correlation
                lr = linregress(est, truth)
                tgt_out['rank_corr_' + key] = round(float(rho), 4)
                tgt_out['inverse_slope_' + key] = round(float(lr.slope), 4)
            tgt_out['reject_soft'] = reject_curve(tgt_out, 'pi_hat_soft',
                                                 [round(0.5 + 0.02 * k, 2) for k in range(1, 16)])
            tgt_out['reject_hard'] = reject_curve(tgt_out, 'pi_hat_hard',
                                                 [round(0.5 + 0.02 * k, 2) for k in range(1, 16)])
            tgt_out['reject_gmm'] = reject_curve(tgt_out, 'pi_hat_gmm',
                                                [round(0.5 + 0.02 * k, 2) for k in range(1, 26)])
            tgt_out['reject_inv'] = reject_curve(tgt_out, 'pi_hat_inv',
                                                [round(0.5 + 0.02 * k, 2) for k in range(1, 26)])
            # 绝对误差：GMM 权重 vs 朴素 AI 比例
            err = {}
            for key in ('pi_hat_soft', 'pi_hat_hard', 'pi_hat_gmm', 'pi_hat_inv'):
                e = [abs(tgt_out['pi_%g' % p][key] - tgt_out['pi_%g' % p]['pi_true'])
                     for p in pis]
                err['mae_' + key] = round(float(np.mean(e)), 4)
                err['max_' + key] = round(float(np.max(e)), 4)
            tgt_out['abs_error'] = err
            log('   %-9s |π̂−π| MAE: soft=%.3f hard=%.3f gmm=%.3f inv=%.3f   最大: %.3f / %.3f / %.3f / %.3f'
                % (tgt, err['mae_pi_hat_soft'], err['mae_pi_hat_hard'],
                   err['mae_pi_hat_gmm'], err['mae_pi_hat_inv'],
                   err['max_pi_hat_soft'], err['max_pi_hat_hard'],
                   err['max_pi_hat_gmm'], err['max_pi_hat_inv']))
            fs_out[tgt] = tgt_out
        out['feature_sets'][fs_name] = fs_out

    out['runtime_sec'] = round(time.time() - t0, 1)
    OUT = Path(args.out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    io.open(OUT, 'w', encoding='utf-8').write(json.dumps(out, ensure_ascii=False, indent=1))
    log('写出 %s  (%.1fs)' % (OUT, out['runtime_sec']))


if __name__ == '__main__':
    main()
