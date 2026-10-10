# -*- coding: utf-8 -*-
"""
TTA 对真实改写攻击的鲁棒性。

改写引入的额外漂移 δ_para 可分解为判别方向分量与正交分量，只有前者进入
Ξ_eff。因此：
  · 若改写主要改变措辞而不改变"像不像 AI"的程度 → Ξ_eff 几乎不变，TTA 增益
    应与无攻击时相当；
  · 若改写把 AI 文本推向人类一侧 → Ξ_eff 增大，可修的漂移更多，TTA 增益
    变大（但伪标签质量同时下降，构成反向拉力）。

攻击实现：Qwen2.5-0.5B-Instruct 本地真实神经改写（非规则替换），与检测器
所用的 chinese-roberta-wwm-ext 无交叉，攻击者与检测器不同源。

条件：
  C0  orig      无攻击：目标 LLM 原文 + 人类原文
  C1  para_ai   仅 AI 侧文本被改写，人类侧保持原样
  C2  para_all  AI 侧与人类侧均被改写（模拟文本统一经润色服务）

指标：macro-F1（none / em）、Ξ_eff（校正前/后）、残留比、E 步伪标签准确率。
输出：results/E33_paraphrase_attack.json
"""
from __future__ import annotations
import io, os, sys, json, time, argparse, importlib.util, random
from pathlib import Path
import numpy as np

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core.invariance import estimate_Xi
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats

import E17_crossllm_ghostbuster as E17

_spec = importlib.util.spec_from_file_location('e26', HERE / 'E26_tta.py')
E26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E26)

N_CG = 800
SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5
MODEL_DIR = HERE / 'models' / 'qwen25_05b'
CORPUS = HERE / 'data' / 'para_attack_corpus.jsonl'
OUT_PATH = HERE / 'results' / 'E33_paraphrase_attack.json'

PARA_PROMPT = ("请把下面这段话改写成意思相同但用词和句式不同的版本。"
               "保持原意和大致长度，不要添加新内容，也不要解释。只输出改写后的文字。\n\n")


def log(*a):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), ' '.join(str(x) for x in a)), flush=True)


# ------------------------------------------------------------------ 阶段 1：生成改写语料
def gen_corpus(targets, n_per, max_new_tokens, threads, batch_size, reseed=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(threads)

    # 断点续跑：已生成的 (key) 跳过
    done = {}
    if CORPUS.exists():
        with CORPUS.open('r', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                if r.get('para'):
                    done[r['key']] = r
        log('  语料已有 %d 条，续跑' % len(done))

    hc3 = E17.build_hc3_index()
    todo = []          # (key, side, target, text)
    for tgt in targets:
        hs, ais, _ = E17.build_pairs(tgt, hc3)
        for i, t in enumerate(ais[:n_per]):
            todo.append(('ai|%s|%d' % (tgt, i), 'ai', tgt, t))
    hs0, _, _ = E17.build_pairs('chatgpt', hc3)
    for i, t in enumerate(hs0[:n_per]):
        todo.append(('human|%d' % i, 'human', '-', t))

    todo = [x for x in todo if x[0] not in done]
    if not todo:
        log('  改写语料已完整，跳过生成')
        return
    # 按原文长度排序后分批：批内长度相近可显著减少解码步数浪费
    todo.sort(key=lambda x: len(x[3]))
    log('  待生成 %d 条（batch=%d, threads=%d, max_new=%d）'
        % (len(todo), batch_size, threads, max_new_tokens))

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR))
    model.eval()
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t0 = time.time()
    with CORPUS.open('a', encoding='utf-8') as fo, torch.no_grad():
        for bi in range(0, len(todo), batch_size):
            chunk = todo[bi:bi + batch_size]
            prompts = [tok.apply_chat_template(
                [{'role': 'user', 'content': PARA_PROMPT + c[3][:200]}],
                tokenize=False, add_generation_prompt=True) for c in chunk]
            enc = tok(prompts, return_tensors='pt', padding=True, truncation=True,
                      max_length=640)
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                 top_p=0.9, temperature=0.8, pad_token_id=tok.pad_token_id)
            for j, c in enumerate(chunk):
                new = out[j][enc['input_ids'].shape[1]:]
                txt = tok.decode(new, skip_special_tokens=True).strip()
                fo.write(json.dumps({'key': c[0], 'side': c[1], 'target': c[2],
                                     'orig': c[3], 'para': txt}, ensure_ascii=False) + '\n')
            fo.flush()
            if (bi // batch_size) % 5 == 0:
                el = time.time() - t0
                rate = (bi + len(chunk)) / max(el, 1e-9)
                log('    %d/%d  %.0fs  (%.2f 条/s, 预计剩余 %.0f 分)'
                    % (bi + len(chunk), len(todo), el, rate,
                       (len(todo) - bi - len(chunk)) / max(rate, 1e-9) / 60))
    log('  改写生成完成 %.0fs' % (time.time() - t0))


def load_corpus():
    d = {}
    if not CORPUS.exists():
        return d
    with CORPUS.open('r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('para'):
                d[r['key']] = r
    return d


# ------------------------------------------------------------------ 阶段 2：评估
def build_condition(corpus, hc3, tgt, cond, n_per):
    """返回 (human_texts, ai_texts)"""
    hs, ais, _ = E17.build_pairs(tgt, hc3)
    hs, ais = hs[:n_per], ais[:n_per]
    if cond == 'orig':
        return list(hs), list(ais)
    if cond == 'para_ai':
        ai = [corpus.get('ai|%s|%d' % (tgt, i), {}).get('para') or ais[i]
              for i in range(len(ais))]
        return list(hs), ai
    if cond == 'para_all':
        ai = [corpus.get('ai|%s|%d' % (tgt, i), {}).get('para') or ais[i]
              for i in range(len(ais))]
        hu = [corpus.get('human|%d' % i, {}).get('para') or hs[i]
              for i in range(len(hs))]
        return hu, ai
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='deepseek,glm')
    ap.add_argument('--n-per', type=int, default=200)
    ap.add_argument('--feats', default='15D,char_2_4')
    ap.add_argument('--conds', default='orig,para_ai,para_all')
    ap.add_argument('--max-new-tokens', type=int, default=170)
    ap.add_argument('--threads', type=int, default=max(1, os.cpu_count() or 4))
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--only-gen', action='store_true')
    ap.add_argument('--only-eval', action='store_true')
    ap.add_argument('--out', default=str(OUT_PATH))
    args = ap.parse_args()

    targets = args.targets.split(',')
    conds = args.conds.split(',')
    want = args.feats.split(',')
    t0 = time.time()

    if not args.only_eval:
        log('阶段 1：生成真实改写语料（Qwen2.5-0.5B-Instruct）')
        gen_corpus(targets, args.n_per, args.max_new_tokens, args.threads, args.batch_size)
    if args.only_gen:
        return

    corpus = load_corpus()
    cov = {}
    for tgt in targets:
        hit = sum(1 for i in range(args.n_per) if ('ai|%s|%d' % (tgt, i)) in corpus)
        cov[tgt] = '%d/%d' % (hit, args.n_per)
    cov_h = sum(1 for i in range(args.n_per) if ('human|%d' % i) in corpus)
    log('语料覆盖：AI 侧 %s；人类侧 %d/%d' % (cov, cov_h, args.n_per))

    log('阶段 2：评估')
    hc3 = E17.build_hc3_index()
    cg_h, cg_a, _ = E17.build_pairs('chatgpt', hc3)
    tr_texts = list(cg_h[:N_CG]) + list(cg_a[:N_CG])
    ytr = np.array([0] * N_CG + [1] * N_CG)

    out = {'protocol': ('攻击者=Qwen2.5-0.5B-Instruct(本地开源, 与检测器不同源) 做真实神经改写；'
                        '检测器=ChatGPT-only(%d) 训练；目标批=目标 LLM 文本(可选改写)；'
                        'TTA=EM 类条件对齐(E26 实现)；5-fold×5seeds；macro-F1' % N_CG),
           'attack_model': 'Qwen2.5-0.5B-Instruct', 'n_per': args.n_per,
           'conds': conds, 'targets': targets, 'feature_sets': {}}

    for fs_name, ctor in E26.FEATURE_SETS:
        if fs_name not in want:
            continue
        log('── %s ──' % fs_name)
        f = ctor()
        # 与 E19/E26 的规范管线一致：中位数填补 + 在训练集上拟合的标准化
        A0 = np.asarray(f.fit(tr_texts).transform(tr_texts), dtype=np.float64)
        imp = SimpleImputer(strategy='median').fit(A0)
        sc = StandardScaler().fit(imp.transform(A0))
        A = sc.transform(imp.transform(A0))
        mu0_tr, mu1_tr = A[ytr == 0].mean(axis=0), A[ytr == 1].mean(axis=0)
        clf = LogisticRegression(max_iter=2000).fit(A, ytr)
        out['feature_sets'][fs_name] = {}

        for tgt in targets:
            out['feature_sets'][fs_name][tgt] = {}
            for cond in conds:
                hs, ais = build_condition(corpus, hc3, tgt, cond, args.n_per)
                te_texts = list(hs) + list(ais)
                yte = np.array([0] * len(hs) + [1] * len(ais))
                B0 = sc.transform(imp.transform(np.asarray(f.transform(te_texts), dtype=np.float64)))

                xi_pre = float(estimate_Xi(A, ytr, B0[yte == 1], B0[yte == 0])['xi_eff'])
                p_none = clf.predict(B0)
                pb_none = clf.predict_proba(B0)[:, 1]
                g_hat = (pb_none > 0.5).astype(int)

                B_em = E26.tta_em(B0, clf, mu0_tr, mu1_tr, 3)
                xi_post = float(estimate_Xi(A, ytr, B_em[yte == 1], B_em[yte == 0])['xi_eff'])
                p_em = clf.predict(B_em)
                pb_em = clf.predict_proba(B_em)[:, 1]

                fold_none, fold_em = [], []
                from sklearn.model_selection import StratifiedKFold
                for seed in SEEDS:
                    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                    for _, idx in skf.split(np.zeros(len(yte)), yte):
                        fold_none.append(float(f1_score(yte[idx], p_none[idx], average='macro')))
                        fold_em.append(float(f1_score(yte[idx], p_em[idx], average='macro')))
                fn, fe = float(np.mean(fold_none)), float(np.mean(fold_em))
                t_s, p_s = stats.ttest_rel(np.array(fold_em), np.array(fold_none))
                # 保守校验：25 个 fold 共享同一数据池，可能低估方差。
                # 按种子聚合（每个 seed 内 5 个 fold 取均值）得 5 个近似独立的配对观测。
                fn_s = np.array(fold_none).reshape(len(SEEDS), -1).mean(axis=1)
                fe_s = np.array(fold_em).reshape(len(SEEDS), -1).mean(axis=1)
                t_seed, p_seed = stats.ttest_rel(fe_s, fn_s)
                cell = {
                    'F1_none': round(fn, 4), 'F1_em': round(fe, 4),
                    'gain': round(fe - fn, 4),
                    'std_none': round(float(np.std(fold_none)), 4),
                    'std_em': round(float(np.std(fold_em)), 4),
                    'p_paired_t': float(p_s),
                    'p_seed_t': float(p_seed),
                    'AUC_none': round(float(roc_auc_score(yte, pb_none)), 4),
                    'AUC_em': round(float(roc_auc_score(yte, pb_em)), 4),
                    'Xi_pre': round(xi_pre, 4), 'Xi_post': round(xi_post, 4),
                    'residual_ratio': (round(xi_post / xi_pre, 4) if abs(xi_pre) > 1e-6 else None),
                    'pseudo_acc': round(float((g_hat == yte).mean()), 4),
                    'n_texts': len(yte),
                }
                out['feature_sets'][fs_name][tgt][cond] = cell
                log('   %-9s %-9s  none=%.4f  em=%.4f  增益=%+.4f  Ξ:%.3f→%.3f(ρ=%.3f)  伪标签acc=%.3f'
                    % (tgt, cond, fn, fe, fe - fn, xi_pre, xi_post,
                       cell['residual_ratio'] or float('nan'), cell['pseudo_acc']))

    out['runtime_sec'] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    io.open(args.out, 'w', encoding='utf-8').write(json.dumps(out, ensure_ascii=False, indent=1))
    log('写出 %s  (%.0fs)' % (args.out, out['runtime_sec']))

    # ---- 汇总表 ----
    print('\n' + '=' * 104)
    print('E33：TTA 对真实改写攻击的鲁棒性')
    print('=' * 104)
    for fs_name in out['feature_sets']:
        print('── %s ──' % fs_name)
        print('%-9s %-9s %8s %8s %8s %9s %9s %8s %8s'
              % ('target', 'cond', 'none', 'em', 'gain', 'Ξ_pre', 'Ξ_post', 'ρ', 'E-acc'))
        for tgt in out['feature_sets'][fs_name]:
            for cond in conds:
                c = out['feature_sets'][fs_name][tgt][cond]
                print('%-9s %-9s %8.4f %8.4f %+8.4f %9.3f %9.3f %8.3f %8.3f'
                      % (tgt, cond, c['F1_none'], c['F1_em'], c['gain'],
                         c['Xi_pre'], c['Xi_post'], c['residual_ratio'] or 0, c['pseudo_acc']))
        print('')


if __name__ == '__main__':
    main()
