# -*- coding: utf-8 -*-
"""
深度表征上的 TTA 验证（特征无关性）。

验证 TTA-EM 是特征无关的算子：不仅能修 15D 手工特征与 TF-IDF 稀疏特征，
也能修预训练语言模型表征的跨 LLM 漂移。

约束：实验环境无 GPU，只做冻结编码器取 embedding（不微调），再在 embedding 上跑
LR + TTA，即深度检测器部署侧常见的「冻结特征 + 轻量头」形态。

模型：hfl/chinese-roberta-wwm-ext（768 维），mean-pooling over last hidden
state（不含 [CLS] 的蒸馏头）。

输出：results/E27_roberta_tta.json
"""
from __future__ import annotations
import io, os, sys, json, time, argparse, importlib.util
from pathlib import Path
import numpy as np

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core.invariance import estimate_Xi

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from scipy import stats

_spec = importlib.util.spec_from_file_location('e26', HERE / 'E26_tta.py')
E26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E26)

_spec2 = importlib.util.spec_from_file_location('e17', HERE / 'E17_crossllm_ghostbuster.py')
E17 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(E17)

SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5
N_CG = 800
OUT_PATH = HERE / 'results' / 'E27_roberta_tta.json'


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


class RobertaFeat:
    """冻结的中文 RoBERTa 表征（mean pooling）"""

    def __init__(self, model_name='hfl/chinese-roberta-wwm-ext', max_len=256, batch=16):
        self.model_name = model_name
        self.max_len = max_len
        self.batch = batch
        self.name = 'roberta_frozen'

    def fit(self, texts):
        import torch
        from transformers import AutoTokenizer, AutoModel
        log('加载 %s ...' % self.model_name)
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        torch.set_num_threads(max(1, os.cpu_count() // 2))
        self.torch = torch
        return self

    @staticmethod
    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def transform(self, texts):
        import torch
        outs = []
        t0 = time.time()
        with torch.no_grad():
            for bi, chunk in enumerate(self._chunks(list(texts), self.batch)):
                enc = self.tok(chunk, truncation=True, max_length=self.max_len,
                               padding=True, return_tensors='pt')
                h = self.model(**enc).last_hidden_state          # (B, L, 768)
                mask = enc['attention_mask'].unsqueeze(-1).float()
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                outs.append(pooled.numpy())
                if bi % 20 == 0:
                    log('   embed %d/%d (%.0fs)' % (bi * self.batch, len(texts), time.time() - t0))
        return np.vstack(outs).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='deepseek,qwen,glm,gpt-4o,llama3')
    ap.add_argument('--mix', default='0,0.2')
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--out', default=str(OUT_PATH))
    args = ap.parse_args()
    OUT = Path(args.out)
    targets = args.targets.split(',')
    alphas = [float(x) for x in args.mix.split(',')]

    t0 = time.time()
    hc3_idx = E17.build_hc3_index()
    data = {}
    for m in ['chatgpt', 'deepseek'] + targets:
        hs, ais, _ = E17.build_pairs(m, hc3_idx)
        data[m] = {'human': hs, 'ai': ais}
        log('  %-10s n=%d' % (m, len(ais)))

    cg, ds = data['chatgpt'], data['deepseek']
    out = {'protocol': ('冻结 chinese-roberta-wwm-ext mean-pooling(768d) + LR 分类头；'
                        'α 混合比例；TTA: none/bn/em；5-fold×5seeds；macro-F1'),
           'model': 'hfl/chinese-roberta-wwm-ext (frozen)', 'max_len': args.max_len,
           'feature_sets': {}}

    fs_out = {}
    for alpha in alphas:
        n_ds = 0 if alpha <= 0 else int(round(N_CG * alpha / (1 - alpha)))
        tr_h = cg['human'][:N_CG] + ds['human'][:n_ds]
        tr_a = cg['ai'][:N_CG] + ds['ai'][:n_ds]
        tr_texts = list(tr_h) + list(tr_a)
        ytr = np.array([0] * len(tr_h) + [1] * len(tr_a))

        feat = RobertaFeat(max_len=args.max_len, batch=args.batch)
        feat.fit(tr_texts)               # 冻结模型，fit 仅加载权重（无标签信息泄漏）
        A = feat.transform(tr_texts)
        log('  train embed %s (%.0fs)' % (A.shape, time.time() - t0))
        sc = StandardScaler().fit(A)
        A = sc.transform(A)
        mu_tr, sd_tr = A.mean(axis=0), A.std(axis=0)
        mu0_tr, mu1_tr = A[ytr == 0].mean(axis=0), A[ytr == 1].mean(axis=0)

        clf = LogisticRegression(max_iter=3000, random_state=42, C=1.0).fit(A, ytr)

        a_out = {'n_ds': n_ds, 'in_domain': {}}
        a_out['in_domain']['train_acc'] = round(float(clf.score(A, ytr)), 4)
        for tgt in targets:
            th, ta = data[tgt]['human'], data[tgt]['ai']
            if tgt == 'deepseek' and n_ds > 0:
                th, ta = th[n_ds:], ta[n_ds:]
            te_texts = list(th) + list(ta)
            yte = np.array([0] * len(th) + [1] * len(ta))
            B0 = sc.transform(feat.transform(te_texts))
            log('  %s embed done (%.0fs)' % (tgt, time.time() - t0))

            cell = {}
            for mode in ['none', 'bn', 'em']:
                if mode == 'none':
                    B = B0
                elif mode == 'bn':
                    B = E26.tta_bn(B0, mu_tr, sd_tr)
                else:
                    B = E26.tta_em(B0, clf, mu0_tr, mu1_tr, 3)
                xi = estimate_Xi(A, ytr, B[yte == 1], B[yte == 0])['xi_eff']
                p = clf.predict(B)
                pb = clf.predict_proba(B)[:, 1]
                fold_f1 = []
                for seed in SEEDS:
                    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                    for _, idx in skf.split(np.zeros(len(yte)), yte):
                        fold_f1.append(float(f1_score(yte[idx], p[idx], average='macro')))
                cell[mode] = {'F1_macro': round(float(np.mean(fold_f1)), 4),
                              'std': round(float(np.std(fold_f1)), 4),
                              'AUC': round(float(roc_auc_score(yte, pb)), 4),
                              'Xi_eff': round(float(xi), 4),
                              'per_fold_F1': [round(x, 4) for x in fold_f1]}
                log('   α=%.2f %-9s %-4s F1=%.4f Ξ=%+.3f (%.0fs)'
                    % (alpha, tgt, mode, np.mean(fold_f1), xi, time.time() - t0))
            base = np.array(cell['none']['per_fold_F1'])
            for mode in ['bn', 'em']:
                cur = np.array(cell[mode]['per_fold_F1'])
                t_s, p_s = stats.ttest_rel(cur, base)
                cell[mode]['delta_vs_none'] = round(float(np.mean(cur - base)), 4)
                cell[mode]['p_paired_t'] = float(p_s)
            cell['theory'] = {
                'Xi_none': cell['none']['Xi_eff'],
                'Xi_bn': cell['bn']['Xi_eff'],
                'Xi_em': cell['em']['Xi_eff'],
                'bn_residual_ratio': (round(cell['bn']['Xi_eff'] / cell['none']['Xi_eff'], 4)
                                      if abs(cell['none']['Xi_eff']) > 1e-6 else None),
                'em_residual_ratio': (round(cell['em']['Xi_eff'] / cell['none']['Xi_eff'], 4)
                                      if abs(cell['none']['Xi_eff']) > 1e-6 else None),
            }
            a_out[tgt] = cell
        fs_out['alpha_%g' % alpha] = a_out
    out['feature_sets']['roberta_frozen'] = fs_out
    out['runtime_sec'] = round(time.time() - t0, 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print('')
    print('=' * 100)
    print('E27：冻结 RoBERTa 表征上的 TTA（特征无关性验证）')
    print('=' * 100)
    for alpha in alphas:
        a = fs_out['alpha_%g' % alpha]
        print('--- α=%.2f  (train acc=%.4f) ---' % (alpha, a['in_domain']['train_acc']))
        print('%-9s %s' % ('target', ''.join('%24s' % m for m in ['none', 'bn', 'em'])))
        for tgt in targets:
            row = '%-9s ' % tgt
            for m in ['none', 'bn', 'em']:
                d = a[tgt][m]
                row += '%24s' % ('%.4f(Ξ%+.3f)' % (d['F1_macro'], d['Xi_eff']))
            print(row)
        print('')
    print('[SAVED] %s  (%.0fs)' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
