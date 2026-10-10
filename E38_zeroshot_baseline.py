# -*- coding: utf-8 -*-
"""
与零样本检测器族的同条件对比。

相关工作（DetectGPT、GLTR、Binoculars 等）依赖白盒概率，而本文的目标域是
闭源商用 LLM，部署方拿不到生成模型的权重，只能用开源替代模型（surrogate）
去算概率——这正是本文主张"泛化能力应在部署侧获取"的动机之一。本实验把这一
主张变成可测的数字：在同一批目标域数据上，用开源替代模型实现零样本检测器族，
与本文方法同协议比较。

实现的基线（全部不接触目标域标签）：
  1. PPL        替代模型的对数困惑度取负，GPTZero 一类的纯概率检测器
  2. PPL/zlib   以 zlib 压缩长度归一化的困惑度，DetectGPT 与 Binoculars 论文
                共同采用的强基线
  3. PLL        掩码语言模型的伪对数似然（单掩码模式一次前向近似）
  4. GLTR       词元秩直方图特征 + 源域（ChatGPT）训练的 logistic 回归
                这是"训练式基线"，与本文检测器同样只见过 ChatGPT，比较最公平

本文方法在同一批样本上重跑（15D / char 2-4gram，无校正与 TTA 两档），保证
样本、协议、指标完全一致。

指标以 AUC 为主（阈值无关、对零样本方法最公平）；另报 F1@源域阈值（真实部署：
阈值只能在源域留出集上定）与 F1@目标域最优阈值（上界）。

输出：results/E38_zeroshot_baseline.json
"""
from __future__ import annotations
import io, os, sys, json, time, math, zlib, argparse, importlib.util
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
from sklearn.metrics import roc_auc_score

_spec = importlib.util.spec_from_file_location('e17', HERE / 'E17_crossllm_ghostbuster.py')
E17 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(E17)
_spec3 = importlib.util.spec_from_file_location('e28', HERE / 'E28_tta_lambda.py')
E28 = importlib.util.module_from_spec(_spec3); _spec3.loader.exec_module(E28)
_spec4 = importlib.util.spec_from_file_location('e29', HERE / 'E29_tta_robust.py')
E29 = importlib.util.module_from_spec(_spec4); _spec4.loader.exec_module(E29)
_spec2 = importlib.util.spec_from_file_location('e26', HERE / 'E26_tta.py')
E26 = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(E26)

N_CG = 800
MAX_LEN = 256
OUT_PATH = HERE / 'results' / 'E38_zeroshot_baseline.json'
_LOCAL_QWEN = HERE / 'models' / 'qwen2.5-0.5b'
CAUSAL = {'gpt2-chinese': 'uer/gpt2-chinese-cluecorpussmall',
          'qwen2.5-0.5b': str(_LOCAL_QWEN) if _LOCAL_QWEN.exists()
                          else 'Qwen/Qwen2.5-0.5B-Instruct'}
MLM = {'roberta-chinese': 'hfl/chinese-roberta-wwm-ext'}


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


def f1_at(y, s, thr, higher_is_ai=True):
    p = ((s > thr) if higher_is_ai else (s < thr)).astype(int)
    return macro_f1(y, p)


def best_f1(y, s, higher_is_ai=True):
    """目标域最优阈值上的 F1（上界，需要标签，仅作参照）。"""
    cand = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 99)))
    return max(f1_at(y, s, t, higher_is_ai) for t in cand)


def summarize(y, s, src_thr, higher_is_ai=True):
    """AUC 为主，另给源域阈值上的 F1 与目标域最优阈值上的 F1。"""
    return {'AUC': round(float(roc_auc_score(y, s)), 4),
            'F1_src_thr': round(f1_at(y, s, src_thr, higher_is_ai), 4),
            'F1_oracle_thr': round(best_f1(y, s, higher_is_ai), 4)}


def pick_src_threshold(s_src_h, s_src_a, higher_is_ai=True):
    """在源域留出集上选使 macro-F1 最优的阈值——零样本方法唯一能做的定阈值方式。"""
    y = np.array([0] * len(s_src_h) + [1] * len(s_src_a))
    s = np.concatenate([s_src_h, s_src_a])
    cand = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 49)))
    best, bt = -1.0, float(np.median(s))
    for t in cand:
        v = f1_at(y, s, t, higher_is_ai)
        if v > best:
            best, bt = v, float(t)
    return bt


def load_causal(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mdl = AutoModelForCausalLM.from_pretrained(CAUSAL[name])
    mdl.eval()
    # 部分中文 GPT-2 仓库的 fast 分词器缺 [UNK]，实测分词会抛错，回退到慢分词器
    try:
        tok = AutoTokenizer.from_pretrained(CAUSAL[name])
        tok.encode('测试中文分词')
    except Exception:
        tok = AutoTokenizer.from_pretrained(CAUSAL[name], use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return mdl, tok


def load_mlm(name):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MLM[name])
    mdl = AutoModelForMaskedLM.from_pretrained(MLM[name])
    mdl.eval()
    return mdl, tok


def batch_nll(mdl, tok, texts, bs=8, max_len=MAX_LEN, with_rank=False):
    """逐条平均负对数似然（自然对数）。返回 (nll, 每条的 token 数[, 秩特征])。

    秩特征（GLTR 用）在同一次前向里顺带算出，避免为它再跑一遍模型——在无 GPU
    的机器上这是能否做完实验的关键。
    """
    import torch
    nlls, ntok, feats = [], [], []
    for i in range(0, len(texts), bs):
        chunk = [str(t)[:max_len * 2] for t in texts[i:i + bs]]
        enc = tok(chunk, return_tensors='pt', padding=True, truncation=True,
                  max_length=max_len)
        ids = enc['input_ids']
        att = enc['attention_mask']
        with torch.no_grad():
            out = mdl(input_ids=ids, attention_mask=att)
            logits = out.logits[:, :-1, :].float()
            tgt = ids[:, 1:]
            # 只取真实词元的对数概率：先 gather 再减 logsumexp，避免把
            # [B,T,V] 的 log_softmax 整体物化（Qwen 词表 15 万，内存吃不消）
            true_logit = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            tok_lp = true_logit - torch.logsumexp(logits, dim=-1)
            mask = att[:, 1:].float()
            s = (tok_lp * mask).sum(dim=1)
            n = mask.sum(dim=1).clamp(min=1)
            ranks = ((logits > true_logit.unsqueeze(-1)).sum(dim=-1).float() + 1.0) \
                if with_rank else None
        nlls.extend((-s / n).tolist())
        ntok.extend(n.tolist())
        if with_rank:
            for j in range(ranks.shape[0]):
                r = ranks[j][mask[j] > 0].double().cpu().numpy()
                p = tok_lp[j][mask[j] > 0].double().cpu().numpy()
                if len(r) == 0:
                    feats.append([0.0] * 6); continue
                feats.append([float((r <= 10).mean()), float((r <= 100).mean()),
                              float((r <= 1000).mean()), float((r > 1000).mean()),
                              float(np.log(r).mean()), float(p.mean())])
        if i == 0 or (i // bs) % 20 == 0:
            log('    前向 %d/%d' % (i + len(chunk), len(texts)))
    if with_rank:
        return np.asarray(nlls), np.asarray(ntok), np.asarray(feats, dtype=np.float64)
    return np.asarray(nlls), np.asarray(ntok)


def batch_rank_feat(mdl, tok, texts, bs=4, max_len=MAX_LEN):
    """GLTR 词元秩特征：每个词元在模型分布中的排名，取分位与直方图。"""
    import torch
    feats = []
    for i in range(0, len(texts), bs):
        chunk = [str(t)[:max_len * 2] for t in texts[i:i + bs]]
        enc = tok(chunk, return_tensors='pt', padding=True, truncation=True,
                  max_length=max_len)
        ids = enc['input_ids']; att = enc['attention_mask']
        with torch.no_grad():
            logits = mdl(input_ids=ids, attention_mask=att).logits.float()
        lg = logits[:, :-1, :]
        tgt = ids[:, 1:]
        true_logit = lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        tok_lp = true_logit - torch.logsumexp(lg, dim=-1)
        # 排名 = 有多少词元的 logit 高于真实词元
        rank = (lg > true_logit.unsqueeze(-1)).sum(dim=-1).float() + 1.0
        m = att[:, 1:].float()
        for j in range(rank.shape[0]):
            r = rank[j][m[j] > 0]
            p = tok_lp[j][m[j] > 0]
            if len(r) == 0:
                feats.append([0.0] * 6); continue
            feats.append([float((r <= 10).mean()), float((r <= 100).mean()),
                          float((r <= 1000).mean()), float((r > 1000).mean()),
                          float(np.log(r).mean()), float(p.mean())])
    return np.asarray(feats, dtype=np.float64)


def batch_pll(mdl, tok, texts, bs=8, max_len=MAX_LEN, mask_frac=0.15, seed=0):
    """单掩码模式的伪对数似然：一次前向估计被掩位置的平均对数概率。"""
    import torch
    rng = np.random.default_rng(seed)
    vals = []
    for i in range(0, len(texts), bs):
        chunk = [str(t)[:max_len * 2] for t in texts[i:i + bs]]
        enc = tok(chunk, return_tensors='pt', padding=True, truncation=True,
                  max_length=max_len)
        ids = enc['input_ids'].clone(); att = enc['attention_mask']
        mask_id = tok.mask_token_id if tok.mask_token_id is not None else 103
        pos = []
        for j in range(ids.shape[0]):
            n = int(att[j].sum().item())
            if n <= 4:
                continue
            cand = [k for k in range(1, n - 1)]
            k = max(1, int(round(mask_frac * len(cand))))
            sel = rng.choice(cand, size=min(k, len(cand)), replace=False)
            ids[j, sel] = mask_id
            pos.append((j, sel))
        with torch.no_grad():
            lp = torch.log_softmax(mdl(input_ids=ids, attention_mask=att).logits.float(),
                                   dim=-1)
        for j, sel in pos:
            orig = enc['input_ids'][j, sel]
            vals.append(float(lp[j, sel, orig].mean().item()))
        # 文本过短的按 0 补齐，保持与 texts 等长
        for j in range(ids.shape[0]):
            if not any(p[0] == j for p in pos):
                vals.append(0.0)
    return np.asarray(vals, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='deepseek,qwen,glm,gpt-4o,llama3')
    ap.add_argument('--n', type=int, default=400, help='每个目标每类取多少条')
    ap.add_argument('--src_n', type=int, default=150, help='源域打分子集每类多少条')
    ap.add_argument('--src_cache', default=str(HERE / 'results' / '_e38_src_cache.npz'),
                    help='源域分数的缓存文件（无 GPU 环境下分批跑完整个实验用）')
    ap.add_argument('--causal', default='gpt2-chinese')
    ap.add_argument('--mlm', default='roberta-chinese')
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--max_len', type=int, default=96)
    ap.add_argument('--skip_mlm', action='store_true', help='跳过掩码模型 PLL 基线')
    ap.add_argument('--no_ours', action='store_true')
    ap.add_argument('--out', default=str(OUT_PATH))
    args = ap.parse_args()

    t0 = time.time()
    targets = args.targets.split(',')
    log('零样本基线对比 E38  目标=%s  n=%d/类  causal=%s  max_len=%d'
        % (targets, args.n, args.causal, args.max_len))

    hc3 = E17.build_hc3_index()
    data = {}
    for m in ['chatgpt'] + targets:
        hs, ais, _ = E17.build_pairs(m, hc3)
        # 源域保留完整：本文检测器要按 N_CG=800/类训练，不能因基线的采样规模被削弱
        if m == 'chatgpt':
            data[m] = {'human': hs, 'ai': ais}
        else:
            data[m] = {'human': hs[:args.n], 'ai': ais[:args.n]}
        log('  %-10s 取 %d+ %d' % (m, len(data[m]['human']), len(data[m]['ai'])))

    cg = data['chatgpt']
    # 基线定阈值与训练 GLTR 用的源域打分子集：LLM 前向是整个实验的瓶颈，独立控规模
    src_n = min(args.src_n, len(cg['human']), len(cg['ai']))
    src_texts = list(cg['human'][:src_n]) + list(cg['ai'][:src_n])
    y_src = np.array([0] * src_n + [1] * src_n)
    log('  源域打分子集 %d+%d（本文检测器仍在 800+800 上训练）' % (src_n, src_n))

    out = {'protocol': ('零样本基线对比。替代模型：因果 %s，掩码 %s。'
                        '目标域每类取 %d 条（与本文方法同一批样本）。'
                        'PPL/PLL/zlib 完全不训练；GLTR 与本文方法同样只使用 ChatGPT 源域'
                        '（%d 组配对）训练。文本截断到前 %d 个词元（无 GPU，'
                        'CPU 前向约 16 词元/秒，长度是唯一可控的开销）。'
                        '阈值一律在源域留出集上按 macro-F1 选定，AUC 为阈值无关主指标。'
                        % (args.causal, ('无' if args.skip_mlm else args.mlm), args.n,
                           N_CG, args.max_len)),
           'n_per_class': args.n, 'causal_model': args.causal,
           'mlm_model': (None if args.skip_mlm else args.mlm),
           'max_len': args.max_len,
           'targets': {}, 'source_thresholds': {}}

    mdl, tok = load_causal(args.causal)
    log('  因果模型载入完成')

    # ── 源域统计量：定阈值与训练 GLTR ──────────────────────────────────
    # 源域打分是最贵的一步且与具体目标无关，落盘缓存，便于按目标分批跑完
    ck = ('src_n=%d;max_len=%d;causal=%s' % (src_n, args.max_len, args.causal))
    cache = Path(args.src_cache)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if str(z['key']) == ck:
            nll_src, F_src = z['nll'], z['feats']
            log('  复用源域缓存 %s（%d 条）' % (cache.name, len(nll_src)))
        else:
            z = None
    else:
        z = None
    if z is None:
        nll_src, _, F_src = batch_nll(mdl, tok, src_texts, bs=args.bs,
                                      max_len=args.max_len, with_rank=True)
        np.savez(cache, key=ck, nll=nll_src, feats=F_src)
        log('  源域打分完成并缓存 -> %s' % cache.name)
    zsrc = np.array([len(zlib.compress(t.encode('utf-8'))) for t in src_texts],
                    dtype=np.float64)
    s_ppl_src = -nll_src                      # 越大越像 AI（PPL 越低越像 AI）
    s_zlib_src = -nll_src / np.maximum(zsrc, 1.0)
    out['source_thresholds']['ppl'] = pick_src_threshold(
        s_ppl_src[y_src == 0], s_ppl_src[y_src == 1])
    out['source_thresholds']['ppl_zlib'] = pick_src_threshold(
        s_zlib_src[y_src == 0], s_zlib_src[y_src == 1])

    mlm_mdl = mlm_tok = None
    if not args.skip_mlm:
        mlm_mdl, mlm_tok = load_mlm(args.mlm)
        pll_src = batch_pll(mlm_mdl, mlm_tok, src_texts, bs=args.bs,
                            max_len=args.max_len)
        s_pll_src = -pll_src
        out['source_thresholds']['pll'] = pick_src_threshold(
            s_pll_src[y_src == 0], s_pll_src[y_src == 1])

    sc_g = StandardScaler().fit(F_src)
    clf_g = LogisticRegression(max_iter=2000, random_state=42, C=1.0).fit(
        sc_g.transform(F_src), y_src)

    # ── 本文方法：源域训练 ─────────────────────────────────────────────
    ours = {}
    if not args.no_ours:
        for fs in ['15D', 'char_2_4']:
            n_tr = min(N_CG, len(cg['human']), len(cg['ai']))
            tr_h, tr_a = list(cg['human'][:n_tr]), list(cg['ai'][:n_tr])
            f, sc, A, clf, mu_tr, sd_tr, mu0_tr, mu1_tr = E29.build_pipe(
                fs, tr_h + tr_a, np.array([0] * len(tr_h) + [1] * len(tr_a)))
            ours[fs] = (f, sc, clf, mu_tr, sd_tr, mu0_tr, mu1_tr)

    for tgt in targets:
        th, ta = data[tgt]['human'], data[tgt]['ai']
        te = list(th) + list(ta)
        y = np.array([0] * len(th) + [1] * len(ta))
        nll, _, F = batch_nll(mdl, tok, te, bs=args.bs, max_len=args.max_len,
                              with_rank=True)
        zl = np.array([len(zlib.compress(t.encode('utf-8'))) for t in te],
                      dtype=np.float64)
        s_ppl = -nll
        s_zlib = -nll / np.maximum(zl, 1.0)
        s_gltr = clf_g.predict_proba(sc_g.transform(F))[:, 1]

        res = {'n': int(len(y)),
               'ppl': summarize(y, s_ppl, out['source_thresholds']['ppl']),
               'ppl_zlib': summarize(y, s_zlib, out['source_thresholds']['ppl_zlib']),
               'gltr': {'AUC': round(float(roc_auc_score(y, s_gltr)), 4),
                        'F1_src_thr': round(macro_f1(y, (s_gltr > 0.5).astype(int)), 4),
                        'F1_oracle_thr': round(best_f1(y, s_gltr), 4)}}
        if mlm_mdl is not None:
            pll = batch_pll(mlm_mdl, mlm_tok, te, bs=args.bs, max_len=args.max_len)
            s_pll = -pll
            res['pll'] = summarize(y, s_pll, out['source_thresholds']['pll'])

        if not args.no_ours:
            for fs, (f, sc, clf, mu_tr, sd_tr, mu0_tr, mu1_tr) in ours.items():
                B0 = np.asarray(f.transform(te), dtype=np.float64) \
                    if not isinstance(f, E28.CachedRoberta) else f.embed('tgt_%s' % tgt, te)
                B0 = sc.transform(np.nan_to_num(B0, nan=0.0, posinf=0.0, neginf=0.0))
                z0 = clf.decision_function(B0)
                Bb = E29._bnc(B0, mu_tr)
                zb = clf.decision_function(Bb)
                res['ours_%s_none' % fs] = {
                    'AUC': round(float(roc_auc_score(y, z0)), 4),
                    'F1_src_thr': round(macro_f1(y, (z0 > 0).astype(int)), 4),
                    'F1_oracle_thr': round(best_f1(y, z0), 4)}
                res['ours_%s_tta' % fs] = {
                    'AUC': round(float(roc_auc_score(y, zb)), 4),
                    'F1_src_thr': round(macro_f1(y, (zb > 0).astype(int)), 4),
                    'F1_oracle_thr': round(best_f1(y, zb), 4)}
        out['targets'][tgt] = res
        log('   %-9s ppl=%.3f  zlib=%.3f  %s gltr=%.3f  |  ' %
            (tgt, res['ppl']['AUC'], res['ppl_zlib']['AUC'],
             ('pll=%.3f  ' % res['pll']['AUC']) if 'pll' in res else '',
             res['gltr']['AUC'])
            + ('  15D %.3f→%.3f  char %.3f→%.3f' % (
                res['ours_15D_none']['AUC'], res['ours_15D_tta']['AUC'],
                res['ours_char_2_4_none']['AUC'], res['ours_char_2_4_tta']['AUC'])
               if not args.no_ours else '')
            + '  (%.0fs)' % (time.time() - t0))

    out['runtime_sec'] = round(time.time() - t0, 1)
    OUT = Path(args.out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # 按目标分批跑时合并进已有结果，而不是覆盖
    if OUT.exists() and not str(OUT).startswith('_'):
        try:
            prev = json.loads(io.open(OUT, encoding='utf-8').read())
            merged = dict(prev)
            merged['targets'].update(out['targets'])
            merged['source_thresholds'].update(out['source_thresholds'])
            merged['runtime_sec'] = prev.get('runtime_sec', 0) + out['runtime_sec']
            out = merged
            log('  合并进已有结果，现有目标：%s' % ', '.join(sorted(out['targets'])))
        except Exception as e:
            log('  合并失败，直接覆盖：%s' % e)
    io.open(OUT, 'w', encoding='utf-8').write(json.dumps(out, ensure_ascii=False, indent=1))
    log('写出 %s  (%.1fs)' % (OUT, out['runtime_sec']))


if __name__ == '__main__':
    main()
