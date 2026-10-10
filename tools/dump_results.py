# -*- coding: utf-8 -*-
"""把 E19/E20/E26/E27/E28/E29/E30 的精确数值导出为 markdown 写作素材。
所有数字直接来自 results/*.json，不做任何人工转写。
"""
import io, json, sys, os
import numpy as np
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
R = 'results'
OUT = []
def w(s=''):
    OUT.append(s)
    print(s)

def L(p):
    return json.load(io.open(os.path.join(R, p), encoding='utf-8'))

def desc():
    w('# 实验数据素材（自动导出，禁止手工改写）')
    w()

# ───────── E19：Ξ_eff 诊断 ─────────
try:
    d = L('E19_invariance_deficit.json')
    w('## E19 Ξ_eff 诊断')
    w()
    if 'summary' in d:
        for k, v in d['summary'].items():
            w('- %s: %s' % (k, json.dumps(v, ensure_ascii=False)[:400]))
        w()
    rows = []
    fs = d.get('feature_sets', d)
    for fs_name, per in fs.items():
        if not isinstance(per, dict):
            continue
        for tgt, cell in per.items():
            if not isinstance(cell, dict):
                continue
            rows.append((fs_name, tgt, cell))
    if rows:
        w('| 特征集 | 目标LLM | Xi_eff | 实测衰减 1-F1 | 其它键 |')
        w('|---|---|---|---|---|')
        for fs_name, tgt, cell in rows[:40]:
            keys = [k for k in cell if isinstance(cell[k], (int, float))]
            w('| %s | %s | %s | %s | %s |' % (
                fs_name, tgt,
                cell.get('Xi_eff', cell.get('xi_eff', '?')),
                cell.get('decay', cell.get('drop', '?')),
                ','.join(keys[:8])))
    w()
except Exception as e:
    w('E19 载入失败: %r' % (e,))

# ───────── E26：TTA 主实验 ─────────
try:
    d = L('E26_tta.json')
    w('## E26 TTA 主实验')
    w()
    w('顶层键: %s' % list(d.keys()))
    if 'meta' in d:
        w('meta: %s' % json.dumps(d['meta'], ensure_ascii=False)[:600])
    w()
    fs = d.get('feature_sets', {})
    for fs_name, per in fs.items():
        w('### 特征集 %s' % fs_name)
        w()
        for alpha, av in (per.items() if isinstance(per, dict) else []):
            if not isinstance(av, dict):
                continue
            w('**混合比例 %s**' % alpha)
            w()
            w('| 目标 | 模式 | F1_macro | Xi_eff | p_paired_t |')
            w('|---|---|---|---|---|')
            for tgt, cell in av.items():
                if not isinstance(cell, dict):
                    continue
                for mode in ['none', 'bn', 'em']:
                    if mode in cell and isinstance(cell[mode], dict):
                        c = cell[mode]
                        w('| %s | %s | %.4f | %.4f | %s |' % (
                            tgt, mode, c.get('F1_macro', float('nan')),
                            c.get('Xi_eff', float('nan')),
                            c.get('p_paired_t', '')))
            w()
except Exception as e:
    w('E26 载入失败: %r' % (e,))

# ───────── E27 RoBERTa ─────────
try:
    d = L('E27_roberta_tta.json')
    w('## E27 RoBERTa 冻结表征 TTA')
    w()
    w('```json')
    w(json.dumps(d, ensure_ascii=False, indent=1)[:3000])
    w('```')
    w()
except Exception as e:
    w('E27 载入失败: %r' % (e,))

# ───────── E28 λ 族 ─────────
for p, tag in [('_E28_i1.json', 'E28 λ族(单步)'), ('_E28_fast.json', 'E28 λ族(fast)')]:
    try:
        d = L(p)
        w('## %s' % tag)
        w()
        w('顶层键: %s' % list(d.keys()))
        if 'meta' in d:
            w('meta: %s' % json.dumps(d['meta'], ensure_ascii=False)[:500])
        fs = d.get('feature_sets', {})
        for fs_name, per in fs.items():
            if not isinstance(per, dict):
                continue
            for alpha, av in per.items():
                if not isinstance(av, dict):
                    continue
                for tgt, cell in av.items():
                    if not isinstance(cell, dict):
                        continue
                    w('%s / %s / %s' % (fs_name, alpha, tgt))
                    for v, c in cell.items():
                        if isinstance(c, dict) and 'F1_macro' in c:
                            w('   %-12s F1=%.4f Xi=%.4f resid=%s%s' % (
                                v, c['F1_macro'], c.get('Xi_eff', float('nan')),
                                c.get('resid', ''),
                                ' oracle' if v == 'oracle' else ''))
        w()
    except Exception as e:
        w('%s 载入失败: %r' % (tag, e))

# ───────── E29 鲁棒 E 步 ─────────
try:
    d = L('_E29_fast.json')
    w('## E29 鲁棒 E 步（15D / char_2_4）')
    w()
    w('顶层键: %s' % list(d.keys()))
    fs = d.get('feature_sets', {})
    for fs_name, per in fs.items():
        if not isinstance(per, dict):
            continue
        for alpha, av in per.items():
            if not isinstance(av, dict):
                continue
            for tgt, cell in av.items():
                if not isinstance(cell, dict):
                    continue
                w('**%s / %s / %s**  (Xi0=%.4f)' % (
                    fs_name, alpha, tgt, cell.get('Xi0', float('nan'))))
                w('| 变体 | F1 | Xi_eff | resid | p | 95%CI |')
                w('|---|---|---|---|---|---|')
                for v, c in cell.items():
                    if isinstance(c, dict) and 'F1_macro' in c:
                        w('| %s | %.4f | %.4f | %s | %s | %s |' % (
                            v, c['F1_macro'], c.get('Xi_eff', float('nan')),
                            c.get('resid', ''), c.get('p_paired_t', ''),
                            c.get('ci95', '')))
                w()
except Exception as e:
    w('E29 载入失败: %r' % (e,))

# ───────── E30 无监督选参 ─────────
for p, tag in [('_E30_gmm+nobal.json', 'E30 选参 GMM+平衡'),
               ('_E30_self.json', 'E30 选参 self'),
               ('_E30_entropy.json', 'E30 选参 entropy'),
               ('_E30_gmm.json', 'E30 选参 gmm')]:
    try:
        d = L(p)
        w('## %s' % tag)
        w()
        w('顶层键: %s' % list(d.keys()))
        for k in ['selection', 'summary', 'auto_select', 'compare']:
            if k in d:
                w('%s: %s' % (k, json.dumps(d[k], ensure_ascii=False)[:2500]))
                w()
        if 'feature_sets' in d:
            for fs_name, per in d['feature_sets'].items():
                if not isinstance(per, dict):
                    continue
                for alpha, av in per.items():
                    if not isinstance(av, dict):
                        continue
                    for tgt, cell in av.items():
                        if not isinstance(cell, dict):
                            continue
                        w('**%s / %s / %s**' % (fs_name, alpha, tgt))
                        for v, c in cell.items():
                            if isinstance(c, dict) and 'F1_obs' in c:
                                w('   %-14s F1obs=%.4f self=%.4f gmm=%.4f stale=%.4f Xi=%.4f imb=%.4f ent=%.4f' % (
                                    v, c.get('F1_obs', 0), c.get('F1_pred_self', 0),
                                    c.get('F1_pred_gmm', 0), c.get('F1_pred_stale', 0),
                                    c.get('Xi_eff', 0), c.get('imbalance', 0),
                                    c.get('entropy', 0)))
                        if 'best_auto' in cell:
                            w('   best_auto=%s (oracle=%s)' % (cell['best_auto'], cell.get('best_true', '')))
                        w()
    except Exception as e:
        w('%s 载入失败: %r' % (tag, e))

# ───────── E20 CLIDA 负面结果 ─────────
try:
    d = L('_E20_v2.json')
    w('## E20 CLIDA（负面结果）')
    w()
    w('顶层键: %s' % list(d.keys()))
    if 'meta' in d:
        w('meta: %s' % json.dumps(d['meta'], ensure_ascii=False)[:800])
    for k, v in d.items():
        if k in ('meta',):
            continue
        if isinstance(v, dict):
            w('### %s' % k)
            w('```json')
            w(json.dumps(v, ensure_ascii=False, indent=1)[:4000])
            w('```')
    w()
except Exception as e:
    w('E20 载入失败: %r' % (e,))

# ───────── E17 跨LLM基线 / E18 in-domain ─────────
for p, tag in [('E17_crossllm_ghostbuster.json', 'E17 跨LLM GHOSTBUSTER 基线'),
               ('E18_indomain_ref.json', 'E18 in-domain 参照')]:
    try:
        d = L(p)
        w('## %s' % tag)
        w()
        w('```json')
        w(json.dumps(d, ensure_ascii=False, indent=1)[:2500])
        w('```')
        w()
    except Exception as e:
        w('%s 载入失败: %r' % (tag, e))

io.open(os.path.join(R, '_ALL_DIGEST.md'), 'w', encoding='utf-8').write('\n'.join(OUT))
print('\n=== 已写出 results/_ALL_DIGEST.md ===')
