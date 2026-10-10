# -*- coding: utf-8 -*-
"""
由 results/*.json 生成稿件表格（markdown）。

用法：
    python tools/make_tables.py                    生成全部表格到 tables/
    python tools/make_tables.py --only 11 18       只生成指定表
    python tools/make_tables.py --ubc em           指定校正口径（默认 oracle）
                                                   em    = E26 迭代类条件对齐
                                                   auto  = E30 无标签自动选择（UBC 定义）
                                                   oracle= E30 真值标签上界

设计目的：稿件中的数字只应存在一份来源（results/*.json）。本脚本是唯一的
"JSON → 表格"通道，正文只写文字，不手写数字。
"""
from __future__ import annotations
import io, os, sys, json, argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / 'results'
OUT = ROOT / 'tables'

TARGETS = ['deepseek', 'qwen', 'glm', 'gpt-4o', 'llama3']
FS_LABEL = {'15D': '15D', 'char_2_4': 'char 2-4'}
COND_LABEL = {'orig': '无攻击', 'para_ai': '仅 AI 侧改写', 'para_all': '双侧均改写'}
FS_LABEL18 = {'15D': '15 维', 'char_2_4': '字符 2-4gram'}


def J(name):
    p = RES / (name + '.json')
    if not p.exists():
        p = RES / '_archive' / (name + '.json')
    with io.open(p, encoding='utf-8') as f:
        return json.load(f)


def r3(x):
    """三位小数，四舍五入（远离零），与稿件排版口径一致。"""
    return str(Decimal(str(float(x))).quantize(Decimal('0.001'),
                                               rounding=ROUND_HALF_UP))


def f3(x):
    return r3(x)


def sg3(x):
    return ('+' if x >= 0 else '-') + r3(abs(x))


def pfmt(p):
    if p >= 0.001:
        return '%.2f' % p
    m, e = ('%.1e' % p).split('e')
    return '$%s\\times10^{%d}$' % (m, int(e))


def holm(ps, alpha=0.05):
    """Holm-Bonferroni：返回每个位置是否显著（输入顺序保持）。"""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    keep = [False] * m
    for rank, i in enumerate(order):
        if ps[i] * (m - rank) <= alpha:
            keep[i] = True
        else:
            break
    return keep


class Table(object):
    def __init__(self, num, title, note=''):
        self.num = num
        self.title = title
        self.note = note
        self.header = []
        self.rows = []

    def render(self):
        out = ['[TABLE %s]' % self.num,
               '| ' + ' | '.join(self.header) + ' |',
               '|' + '|'.join(['---'] * len(self.header)) + '|']
        for r in self.rows:
            out.append('| ' + ' | '.join(r) + ' |')
        if self.note:
            out.append('')
            out.append('*注：%s*' % self.note)
        return '\n'.join(out) + '\n'


def fnum(x, nd=4):
    """固定小数位后去掉尾部多余的 0（0.4990 → 0.499，0.0295 保持四位）。"""
    return ('%.*f' % (nd, x)).rstrip('0').rstrip('.')


# ── 表 8：单因子 vs 双因子对实测精度的解释力 ───────────────────────────────
def t8():
    # 表 8 的回归样本为 3 目标版本（deepseek,qwen,glm），与表 10 同口径；
    # 5 目标版本的 predictive 段见 results/E30_tta_select.json。
    d = J('_E30_pred3')
    p = d.get('predictive') or {}
    sgl, dbl = p.get('single', {}), p.get('double', {})
    tb = Table(8, '表 8')
    tb.header = ['模型', '自变量', '$R^2$', 'MAE']
    tb.rows.append(['单因子', '仅残留漂移 $\\Xi^{\\mathrm{post}}_{\\mathrm{eff}}$',
                    fnum(sgl.get('R2', 0)), fnum(sgl.get('MAE', 0))])
    tb.rows.append(['**双因子**',
                    '残留漂移 + 注入方差 $\\sigma^2_{\\mathrm{inj}}$',
                    '**%s**' % fnum(dbl.get('R2', 0)),
                    '**%s**' % fnum(dbl.get('MAE', 0))])
    tb.note = ('在 E30 的 %s 个"变体×目标"样本点上回归 macro-F1 的实测值，'
               '覆盖 deepseek、qwen、glm 三个目标 LLM；数据源 results/_E30_pred3.json。'
               % p.get('n_variants', '—'))
    return tb


# ── 表 9：校正族三维对比（BN / EM / conf_k / oracle） ──────────────────────
def t9():
    e29 = J('_E29_fast')
    e29r = J('_E29_roberta')
    tb = Table(9, '表 9')
    tb.header = ['特征集 ($d$)', '目标', '无校正', 'BN', 'EM',
                 '$\\mathrm{conf}_{k^*}$', '$k^*$', 'oracle（真值标签）',
                 '$\\rho_{\\mathrm{EM}}$']
    panels = [(e29, '15D', '15D', 15), (e29, 'char_2_4', 'char 2-4', 50),
              (e29r, 'roberta', 'RoBERTa', 768)]
    for src, key, lab, dim in panels:
        a = src['feature_sets'][key]['alpha_0']
        for t in ['deepseek', 'glm']:
            c = a[t]
            confs = [(k, v['F1_macro']) for k, v in c.items() if k.startswith('conf_')]
            kbest = max(confs, key=lambda kv: kv[1])
            vals = [c['none']['F1_macro'], c['bn']['F1_macro'],
                    c['em']['F1_macro'], kbest[1]]
            bi = max(range(len(vals)), key=lambda i: (vals[i], i))

            def m(i):
                return '**%s**' % r3(vals[i]) if i == bi else r3(vals[i])
            tb.rows.append(['%s (%d)' % (lab, dim), t, m(0), m(1), m(2), m(3),
                            '%g' % float(kbest[0].split('_')[1]),
                            r3(c['oracle']['F1_macro']), r3(c['em']['resid'])])
    tb.note = ('$k^*$ 为使 macro-F1 最大的置信门控比例（候选 $k\\in\\{0.3,0.5,0.7\\}$，'
               '并列时取较小者）；$\\rho_{\\mathrm{EM}}$ 为 EM 校正后的残留漂移比 '
               '$\\Xi^{\\mathrm{post}}_{\\mathrm{eff}}/\\Xi^{\\mathrm{pre}}_{\\mathrm{eff}}$。')
    return tb


# ── 表 10：零标签选参准则的平均损失 ────────────────────────────────────────
CRIT_FILES = [('_E30_fast', '陈旧伪标签（未更新）'),
              ('_E30_gmm', '一维两分量 GMM'),
              ('_E30_self', '校正后自标签'),
              ('_E30_entropy', '最小预测熵'),
              ('_E30_gmm+nobal', 'GMM + 类平衡先验')]


def t10():
    rows = []
    for name, label in CRIT_FILES:
        d = J(name)
        losses = []
        for fs in ['15D', 'char_2_4']:
            a = d['feature_sets'][fs]['alpha_0']
            for t, v in a.items():
                if not t.endswith('_choice'):
                    continue
                losses.append(v['oracle_F1'] - v['auto_F1'])
        rows.append((label, sum(losses) / len(losses), len(losses)))
    rows.sort(key=lambda r: -r[1])
    best = min(r[1] for r in rows)
    tb = Table(10, '表 10')
    tb.header = ['选择准则', '平均 F1 损失（相对 oracle 选参）', '是否需要标签']
    for label, loss, n in rows:
        tb.rows.append(['**%s**' % label if abs(loss - best) < 1e-9 else label,
                        '%.4f' % loss, '否'])
    ns = sorted(set(r[2] for r in rows))
    tb.note = ('平均损失在 %s 组配置（2 特征集 × 目标 LLM）上计算，五种准则覆盖'
               '同一组配置。' % ('、'.join(str(x) for x in ns)))
    return tb


# ── 表 11：零异源数据主结果 ────────────────────────────────────────────────
def t11(ubc='auto'):
    e26 = J('E26_tta')
    e30 = J('E30_tta_select') if (RES / 'E30_tta_select.json').exists() else None
    cells = []
    for fs in ['15D', 'char_2_4']:
        a = e26['feature_sets'][fs]['alpha_0']
        for t in TARGETS:
            c = a[t]
            p = c['em']['p_paired_t']
            if ubc == 'em':
                corr = c['em']['F1_macro']
            elif e30 is not None:
                ch = e30['feature_sets'][fs]['alpha_0'].get(t + '_choice', {})
                corr = ch.get({'auto': 'auto_F1', 'oracle': 'oracle_F1'}[ubc],
                              c['em']['F1_macro'])
            else:
                corr = c['em']['F1_macro']
            cells.append([fs, t, c['none']['F1_macro'], c['bn']['F1_macro'], corr,
                          corr - c['none']['F1_macro'], c['none']['AUC'], p])
    keep = holm([r[7] for r in cells])
    tb = Table(11, '表 11')
    tb.header = ['特征集', '目标 LLM', '无校正', '+BN', '+EM/UBC',
                 '$\\Delta$（相对无校正）', 'AUC', '显著']
    for r, k in zip(cells, keep):
        best = max(r[2], r[3], r[4])
        def mark(v):
            return '**%s**' % f3(v) if abs(v - best) < 1e-9 else f3(v)
        tb.rows.append([FS_LABEL[r[0]], r[1], mark(r[2]), mark(r[3]), mark(r[4]),
                        sg3(r[5]), f3(r[6]), '✓' if k else ''])
    tb.note = ('$\\alpha=0$，5-fold×5 seeds，macro-F1。"显著"指配对 $t$ 检验经 Holm '
               '校正后 $p<0.05$。校正口径=%s。' % ubc)
    return tb


# ── 表 12：增益随初始 Ξ 的分档 ─────────────────────────────────────────────
def t12():
    e26 = J('E26_tta')
    groups = []
    for fs in ['15D', 'char_2_4']:
        for al in ['alpha_0', 'alpha_0.05', 'alpha_0.2']:
            a = e26['feature_sets'][fs][al]
            for t in TARGETS:
                c = a[t]
                groups.append((c['none']['Xi_eff'], c['em']['delta_vs_none'], fs, t, al))
    bins = [(None, 0.2, '$\\leq 0.2$'), (0.2, 0.6, '$0.2\\sim0.6$'),
            (0.6, 1.2, '$0.6\\sim1.2$'), (1.2, None, '$>1.2$')]
    tb = Table(12, '表 12')
    tb.header = ['初始 $\\Xi_{\\mathrm{eff}}$', '组数', '平均增益', '代表场景']
    for lo, hi, lab in bins:
        sel = [g for g in groups if (lo is None or g[0] > lo) and (hi is None or g[0] <= hi)]
        if not sel:
            continue
        mean = sum(g[1] for g in sel) / len(sel)
        rep = max(sel, key=lambda g: g[1])
        tb.rows.append([lab, str(len(sel)),
                        '**%s**' % sg3(mean) if mean >= 0.03 else sg3(mean),
                        '%s / %s' % (FS_LABEL[rep[2]], rep[3])])
    tb.note = '分档由 30 组配置（2 特征集 × 5 目标 × 3 混合比例）的初始 $\\Xi_{\\mathrm{eff}}$ 划分。'
    return tb


# ── 表 13 / 14：与训练端混合的对比 ─────────────────────────────────────────
def t13(ubc='auto'):
    e26 = J('E26_tta')
    e30 = J('E30_tta_select') if (RES / 'E30_tta_select.json').exists() else None
    rows = []
    for fs in ['15D', 'char_2_4']:
        for t in ['deepseek', 'glm']:
            a0 = e26['feature_sets'][fs]['alpha_0'][t]
            a5 = e26['feature_sets'][fs]['alpha_0.05'][t]
            a2 = e26['feature_sets'][fs]['alpha_0.2'][t]
            if ubc == 'em':
                corr = a0['em']['F1_macro']
            elif e30 is not None:
                ch = e30['feature_sets'][fs]['alpha_0'].get(t + '_choice', {})
                corr = ch.get({'auto': 'auto_F1', 'oracle': 'oracle_F1'}[ubc],
                              a0['em']['F1_macro'])
            else:
                corr = a0['em']['F1_macro']
            rows.append([FS_LABEL[fs], t, a0['none']['F1_macro'], a5['none']['F1_macro'],
                         a2['none']['F1_macro'], corr, corr - a2['none']['F1_macro']])
    tb = Table(13, '表 13')
    tb.header = ['特征集', '目标', '① 无处理', '② 混合 5%', '③ 混合 20%',
                 '④ $\\alpha{=}0$ + UBC', '④ vs ③']
    for r in rows:
        tb.rows.append([r[0], r[1], f3(r[2]), f3(r[3]), f3(r[4]),
                        '**%s**' % f3(r[5]), sg3(r[6])])
    tb.note = '① ②③④ 均为 macro-F1，5-fold×5 seeds；④ 的校正口径=%s。' % ubc
    return tb


def t14(ubc='em'):
    e26 = J('E26_tta')
    tb = Table(14, '表 14')
    tb.header = ['特征集', '目标', '$\\alpha{=}0.2$', '$\\alpha{=}0.2$ + UBC', '增益']
    for fs in ['15D', 'char_2_4']:
        for t in ['deepseek', 'glm']:
            a = e26['feature_sets'][fs]['alpha_0.2'][t]
            corr = a['em']['F1_macro']
            tb.rows.append([FS_LABEL[fs], t, f3(a['none']['F1_macro']),
                            '**%s**' % f3(corr), sg3(corr - a['none']['F1_macro'])])
    tb.note = '混合训练与部署侧校正叠加；校正口径=E26 迭代类条件对齐。'
    return tb


# ── 表 15：无标签已实现的比例（Ceiling 对照） ──────────────────────────────
def t15(ubc='oracle'):
    e26 = J('E26_tta')
    e30 = J('E30_tta_select')
    e29 = J('_E29_fast')
    e29r = J('_E29_roberta')
    tb = Table(15, '表 15')
    tb.header = ['特征集', '目标', '无校正', 'UBC（伪标签）', 'Ceiling（真值标签）',
                 '已实现的比例']
    for fs in ['15D', 'char_2_4']:
        a26 = e26['feature_sets'][fs]['alpha_0']
        a30 = e30['feature_sets'][fs]['alpha_0']
        a29 = e29['feature_sets'][fs]['alpha_0']
        for t in ['deepseek', 'glm']:
            none = a26[t]['none']['F1_macro']
            if ubc == 'em':
                corr = a26[t]['em']['F1_macro']
            else:
                key = {'auto': 'auto_F1', 'oracle': 'oracle_F1'}[ubc]
                corr = a30.get(t + '_choice', {}).get(key, a26[t]['em']['F1_macro'])
            ceil = a29[t]['oracle']['F1_macro']
            pct = (corr - none) / (ceil - none) if abs(ceil - none) > 1e-9 else 0.0
            tb.rows.append([FS_LABEL[fs], t, r3(none), r3(corr), r3(ceil),
                            '%.0f%%' % (100 * pct)])
    a = e29r['feature_sets']['roberta']['alpha_0']
    for t in ['deepseek', 'glm']:
        c = a[t]
        confs = [(k, v['F1_macro']) for k, v in c.items() if k.startswith('conf_')]
        corr = max(confs, key=lambda kv: kv[1])[1]
        none = c['none']['F1_macro']
        ceil = c['oracle']['F1_macro']
        pct = (corr - none) / (ceil - none)
        tb.rows.append(['RoBERTa', t, r3(none), r3(corr), r3(ceil),
                        '%.0f%%' % (100 * pct)])
    tb.note = ('已实现的比例 $=(\\mathrm{UBC}-\\text{无校正})/(\\mathrm{Ceiling}-'
               '\\text{无校正})$，由未取整的实测值计算；UBC 口径=%s。' % ubc)
    return tb


# ── 表 17：λ 消融（类条件对齐单步实现） ────────────────────────────────────
def t17():
    e28 = J('_E28_i1')
    tb = Table(17, '表 17')
    tb.header = ['特征集', '目标', '$\\lambda{=}0$', '0.25', '0.5', '0.75', '1.0']
    for fs in ['15D', 'char_2_4']:
        a = e28['feature_sets'][fs]['alpha_0']
        for t in ['deepseek', 'glm']:
            c = a[t]
            vals = [c['lam_%s' % s]['F1_macro']
                    for s in ['0', '0.25', '0.5', '0.75', '1']]
            bi = max(range(len(vals)), key=lambda i: (vals[i], i))  # 并列取最右
            tb.rows.append([FS_LABEL[fs], t] +
                           ['**%s**' % r3(v) if i == bi else r3(v)
                            for i, v in enumerate(vals)])
    tb.note = '迭代次数固定为 1、位移减批均值；不含置信门控变体。'
    return tb


# ── 表 18：批次规模敏感性 ──────────────────────────────────────────────────
def t18():
    e31 = J('E31_batch_size')
    e31s = J('E31_small_batch')      # n<50 的补充抽样（独立随机流，不污染原序列）
    e32 = J('E32_roberta_batch')
    sizes = ['10', '20', '30', '40', '50', '100', '200', '500', '1000', '2000']
    tb = Table(18, '表 18')
    tb.header = ['批次规模 $n$'] + sizes
    for src, ssrc, fs, lab in [(e31, e31s, '15D', '15D / %s'),
                               (e31, e31s, 'char_2_4', 'char 2-4 / %s'),
                               (e32, None, 'roberta_frozen', 'RoBERTa / %s')]:
        for t in ['deepseek', 'glm']:
            node = src['feature_sets'][fs][t]
            snode = ssrc['feature_sets'][fs][t] if ssrc else {}
            row = [lab % t]
            for s in sizes:
                k = 'n_' + s
                if k in node:
                    row.append(sg3(node[k]['gain']))
                elif ssrc and k in snode:
                    row.append(sg3(snode[k]['gain']))
                else:
                    row.append('—')
            tb.rows.append(row)
    tb.note = ('目标批 Bootstrap 重采样到 $n$ 后独立执行校正；数值为相对无校正的 macro-F1 '
               '增益，5-fold×5 seeds。$n\\geq50$ 六列与本文其余实验共用同一随机流；'
               '$n\\leq40$ 四列为补充抽样（独立随机流，未改动前六列），'
               'RoBERTa 列未补测小规模档位。')
    return tb


# ── 表 19：类先验失衡下的校正增益 ──────────────────────────────────────────
def t19():
    e35 = J('E35_prior_scan')
    pis = ['%g' % float(p) for p in e35['pis']]
    tb = Table(19, '表 19')
    tb.header = ['特征集 / 目标'] + ['$\\pi$=%.1f' % float(p) for p in pis]
    for fs in ['15D', 'char_2_4']:
        for t in ['deepseek', 'glm']:
            node = e35['feature_sets'][fs][t]
            row = ['%s / %s' % (FS_LABEL[fs], t)]
            for p in pis:
                c = node['pi_' + p]
                if c['revert_rate'] >= 0.8:
                    row.append('退回')            # 类平衡先验判所有候选无效
                elif c['worse_rate'] >= 0.5:
                    row.append('**%s**$^{\\dagger}$' % sg3(c['gain']))
                else:
                    row.append(sg3(c['gain']))
            tb.rows.append(row)
    tb.note = ('按 AI 占比 $\\pi$ 抽样目标批（总数 %d，每格 %d 次独立抽样），'
               '数值为 auto 相对无校正的 macro-F1 增益均值；'
               '「退回」表示类平衡先验在 $\\geq80\\%%$ 的重复中把所有候选判为无效、'
               '系统退回不校正；$^{\\dagger}$ 表示 $\\geq50\\%%$ 的重复中出现负增益。'
               % (e35['batch_size'], e35['reps']))
    return tb


# ── 表 20：失衡下的 AI 类召回与告警行为 ────────────────────────────────────
def t20():
    e35 = J('E35_prior_scan')
    pis = ['%g' % float(p) for p in e35['pis']]
    tb = Table(20, '表 20')
    tb.header = ['特征集 / 目标', '指标'] + ['$\\pi$=%.1f' % float(p) for p in pis]
    for fs in ['15D', 'char_2_4']:
        for t in ['deepseek', 'glm']:
            node = e35['feature_sets'][fs][t]
            for key, lab, fmt in [
                    ('recall', 'AI 召回（无校正→auto）',
                     lambda c: '%.3f→%.3f' % (c['recall_none'], c['recall_auto'])),
                    ('imbalance', '$\\hat w$ 偏离 $0.5$ 的幅度（无校正）',
                     lambda c: r3(c['imbalance_none'])),
                    ('nvalid', '有效候选数',
                     lambda c: '%.1f' % c['n_valid_candidates']),
                    ('revert', '退回率',
                     lambda c: '%.0f\\%%' % (100 * c['revert_rate']))]:
                tb.rows.append(['%s / %s' % (FS_LABEL[fs], t), lab]
                               + [fmt(node['pi_' + p]) for p in pis])
    tb.note = ('AI 召回为 AI 类样本被判为 AI 的比例（$\\pi\\to1$ 时批次中没有人类样本，'
               'macro-F1 退化为 AI 类 F1 的一半而不可横向比较，此时以 AI 召回为准）；'
               '有效候选数为通过类平衡先验（阈值 %.2f）的候选变体个数（满额 11）。'
               % e35['imbalance_tol'])
    return tb


# ── 表 21：真实改写攻击 ────────────────────────────────────────────────────
def t21():
    diag = J('E33b_paraphrase_diag')
    atk = J('E33_paraphrase_attack')
    tb = Table(21, '表 21')
    tb.header = ['特征集', '目标', '条件', '未校正', 'TTA-EM', '增益', '$p$',
                 '阈值上界', '上界捕获率', '$\\Delta_T$', '$\\Xi_{\\mathrm{ai}}$',
                 '$\\Xi_{\\mathrm{hu}}$']
    for fs in ['15D', 'char_2_4']:
        for t in ['deepseek', 'glm']:
            for cond in ['orig', 'para_ai', 'para_all']:
                d = diag['feature_sets'][fs]['targets'][t][cond]
                a = atk['feature_sets'][fs][t][cond]
                none_f1, em_f1, top = d['F1_none'], d['F1_em'], d['F1_optthr']
                cap = (em_f1 - none_f1) / (top - none_f1) if abs(top - none_f1) > 1e-9 else 0.0
                tb.rows.append([FS_LABEL18[fs], t, COND_LABEL[cond], f3(none_f1),
                                f3(em_f1), sg3(em_f1 - none_f1), pfmt(a['p_paired_t']),
                                f3(top), '%.2f' % cap, f3(d['Delta_T']),
                                sg3(d['xi_ai']), sg3(d['xi_hu'])])
    tb.note = ('目标批每格 200 篇人类 + 200 篇 AI；F1 为目标批整体口径；$p$ 为 5-fold×5 '
               'seeds 配对 $t$ 检验；阈值上界由真值标签选取。')
    return tb


# ── 表 26 / 20：CLIDA 与逐维变体 ──────────────────────────────────────────
def t26():
    e20 = J('_E20_v2')
    tb = Table(26, '表 26')
    tb.header = ['特征集', '目标', '$\\gamma{=}0$（普通 Fisher）', '$\\gamma{=}1$',
                 '$\\gamma{=}3$', '原始 LR 基线']
    for fs in ['15D', 'char_2_4']:
        b = e20['feature_sets'][fs]['alpha_0']
        for t in ['deepseek', 'glm']:
            tb.rows.append([FS_LABEL[fs], t] +
                           [r3(b['gamma_%s' % g][t]['clida']['F1_macro'])
                            for g in ('0', '1', '3')] +
                           [r3(b['gamma_0'][t]['raw_lr']['F1_macro'])])
    tb.note = ('$\\alpha=0$，5-fold×5 seeds，macro-F1。$S_E$ 已归一到与 $S_W$ 同 '
               'trace 尺度。加粗为讨论性标注，不由本生成器决定。')
    return tb


def t27():
    e20 = J('_E20_v2')
    tb = Table(27, '表 27')
    tb.header = ['特征集', '目标', '原始 LR', '`diag_heur`（本文前期调制）',
                 '`diag_fisher`', 'CLIDA $\\gamma{=}0$']
    for fs in ['char_2_4', '15D']:
        b = e20['feature_sets'][fs]['alpha_0']
        for t in ['deepseek', 'glm']:
            c = b['gamma_0'][t]
            tb.rows.append([FS_LABEL[fs], t,
                            r3(c['raw_lr']['F1_macro']),
                            r3(c['diag_heur']['F1_macro']),
                            r3(c['diag_fisher']['F1_macro']),
                            r3(c['clida']['F1_macro'])])
    tb.note = '$\\alpha=0$，5-fold×5 seeds，macro-F1。'
    return tb


# ── 表 28：S_E 与 S_B 的重合程度 ───────────────────────────────────────────
def t28():
    d = J('E21_se_sb_angle')
    tb = Table(28, '表 28')
    tb.header = ['特征集', '$\\mathrm{tr}(S_B)$', '$\\mathrm{tr}(S_W)$',
                 '$\\mathrm{tr}(S_E)$', '$\\mathrm{tr}(S_E)/\\mathrm{tr}(S_W)$',
                 '主方向余弦', '前 3 主方向最大主余弦',
                 '$\\mathrm{tr}(S_E)/\\mathrm{tr}(S_B)$']
    for fs in ['15D', 'char_2_4']:
        c = d['feature_sets'][fs]
        tb.rows.append([FS_LABEL[fs], r3(c['tr_S_B']), r3(c['tr_S_W']),
                        r3(c['tr_S_E']), r3(c['ratio_SE_SW']),
                        r3(c['cos_top1']), '**%s**' % r3(c['cos_top3']),
                        r3(c['ratio_SE_SB'])])
    tb.note = ('$\\alpha=%s$，环境为 %s，散度在标准化后的训练集上估计；'
               '数据源 tools/se_sb_angle.py。' % (d['alpha'], d['env']))
    return tb


# ── 表 22：误报约束下的工作点（E36）─────────────────────────────────────
def t22(ubc='oracle'):
    e36 = J('E36_fpr_metrics')
    tb = Table(22, '表 22')
    tb.header = ['特征集 / 目标', 'F1 无校正', 'F1 校正', 'AUC 无校正', 'AUC 校正',
                 'TPR@FPR=1% 无→校', 'TPR@FPR=5% 无→校',
                 'FPR@$z$=0 无校正', 'FPR@$z$=0 校正',
                 'ECE 无校正', 'ECE 校正']
    max_dauc = 0.0
    max_drho = 0.0
    ece_up = 0
    for fs in ['15D', 'char_2_4']:
        for t in TARGETS:
            node = e36['feature_sets'][fs][t]
            v = node['chosen'].get(ubc) or 'none'
            n0, g = node['cells']['none'], node['cells'][v]
            max_dauc = max(max_dauc, abs(g['auc_delta']))
            max_drho = max(max_drho, abs(1.0 - g['spearman_vs_none']))
            if g['ECE'] > n0['ECE']:
                ece_up += 1
            tb.rows.append([
                '%s / %s' % (FS_LABEL[fs], t), f3(n0['F1']), f3(g['F1']),
                f3(n0['AUC']), f3(g['AUC']),
                '%s → %s' % (f3(n0['TPR@FPR=0.01']), f3(g['TPR@FPR=0.01'])),
                '%s → %s' % (f3(n0['TPR@FPR=0.05']), f3(g['TPR@FPR=0.05'])),
                f3(n0['FPR_z0']), f3(g['FPR_z0']),
                f3(n0['ECE']), f3(g['ECE'])])
    tb.note = ('目标批为该目标 LLM 全部人机配对文本（$n$=2000）；宏观 F1 与表 11 同为 '
               '5-fold×5 seeds 折外均值（复算与存档逐格一致，最大偏差 0.000）；'
               'TPR@FPR 在整批上估计并给出 %d 次 Bootstrap 标准误。'
               '类条件对齐后斯皮尔曼 $\\rho$ 与无校正分数的最大偏离为 %.4f、'
               'AUC 最大变化 %.4f，即校正在分数上是（近）严格单调重标定：'
               '**TPR@FPR 不变，收益全部落在 $z$=0 这一决策阈值上**。'
               'ECE 为按置信度 10 箱计算的期望校准误差（10 组中 %d 组校正后变差，'
               '即阈值被推到正确位置并不改善概率校准）。校正口径=%s。'
               % (e36.get('n_boot', 0), max_drho, max_dauc, ece_up, ubc))
    return tb


# ── 表 24：无标签条件下能否命中误报预算（E36）─────────────────────────────
def t24(ubc='oracle'):
    e36 = J('E36_fpr_metrics')
    tb = Table(24, '表 24')
    tb.header = ['特征集 / 目标'] + ['$\\alpha$=%d%% 无校正' % int(100 * a)
                                     for a in (0.01, 0.05, 0.10)] \
                                 + ['$\\alpha$=%d%% 校正' % int(100 * a)
                                    for a in (0.01, 0.05, 0.10)]
    errs = {'none': {}, 'tta': {}}
    for fs in ['15D', 'char_2_4']:
        for t in TARGETS:
            node = e36['feature_sets'][fs][t]
            v = node['chosen'].get(ubc) or 'none'
            fb = node['fpr_budget']
            row = ['%s / %s' % (FS_LABEL[fs], t)]
            for vv in ['none', v]:
                for a in ('0.01', '0.05', '0.1'):
                    c = fb[vv]['a=%s' % a]
                    row.append('%s (%s)' % (f3(c['FPR_realized']), f3(c['TPR'])))
                    errs['none' if vv == 'none' else 'tta'].setdefault(a, []).append(
                        abs(c['FPR_realized'] - float(a)))
            tb.rows.append(row)
    # oracle 变体可能就是 none（如 char 2-4/gpt-4o），此时校正侧没有独立误差项
    mae = {k: {a: (sum(errs[k].get(a, [0.0])) / len(errs[k].get(a, [0.0])))
                  for a in ('0.01', '0.05', '0.1')} for k in ('none', 'tta')}
    tb.note = ('阈值取源域（ChatGPT 留出）人类分数的 $(1-\\alpha)$ 分位数——该量在部署前'
               '即可获得，不需要任何目标域标签。单元格为「实际实现的 FPR（对应 TPR）」。'
               '平均命中误差：无校正 %s，校正后 %s（$\\alpha$ 取 1%%/5%%/10%%）。'
               % ('/'.join(f3(mae['none'][a]) for a in ('0.01', '0.05', '0.1')),
                  '/'.join(f3(mae['tta'][a]) for a in ('0.01', '0.05', '0.1'))))
    return tb


# ── 表 16：目标域监督上界与恢复率（E37）──────────────────────────────────
def t16():
    e37 = J('E37_supervised_ceiling')
    tb = Table(16, '表 16')
    tb.header = ['特征集 / 目标', '无校正', '本文·无标签选参', '本文·真值选参',
                 '目标域监督上界', '总缺口', '恢复率·无标签', '恢复率·真值']
    rr, rra, auc_gap = [], [], []
    for fs in ['15D', 'char_2_4']:
        for t in TARGETS:
            c = e37['feature_sets'][fs][t]
            if c.get('degenerate_gap'):
                rr.append(None)
            else:
                rr.append(c['recovery_rate']); rra.append(c['recovery_rate_auto'])
            auc_gap.append(c['AUC_sup'] - c['AUC_none'])
            def rr_fmt(v):
                return '—' if v is None else f3(v)
            tb.rows.append([
                '%s / %s' % (FS_LABEL[fs], t), f3(c['F1_none']),
                f3(c['F1_tta_auto']), f3(c['F1_tta']),
                '**%s**' % f3(c['F1_sup']), sg3(c['gap_total']),
                rr_fmt(c['recovery_rate_auto']), rr_fmt(c['recovery_rate'])])
    ok = [v for v in rr if v is not None]
    auc_mv = max(abs(c['AUC_tta_auto'] - c['AUC_none'])
                 for fs in ['15D', 'char_2_4'] for t in TARGETS
                 for c in [e37['feature_sets'][fs][t]])
    tb.note = ('目标域内 5-fold×5 seeds 交叉验证，每折以 %d+%d 条目标域样本训练、'
               '留出折测试；无校正臂用同规模 ChatGPT-only 训练集在同一批留出折上测试；'
               'TF-IDF 词表与标准化参数只在训练折内拟合。'
               '总缺口 = 监督上界 − 无校正，恢复率 = （本文 − 无校正）/ 总缺口。'
               '无标签选参 = 一维 GMM 准则自动挑变体（部署实际形态），真值选参 = 折内 '
               'oracle（选参已用到标签，仅作参照）。缺口绝对值 < 0.01 的配置恢复率无意义，'
               '记「—」。有效配置的恢复率均值：无标签 %.3f、真值 %.3f（%d 组）。'
               '监督上界的 AUC 平均比无校正高 %.3f，而本文校正（无标签选参）的 AUC '
               '变化不超过 %.3f——残余段是排序/表示层面的损失，无法由无标签校正确取得。'
               % (e37['n_train_per_class'], e37['n_train_per_class'],
                  sum(rra) / len(rra), sum(ok) / len(ok), len(ok),
                  sum(auc_gap) / len(auc_gap), auc_mv))
    return tb


# ── 表 23：整批对齐会损害排序，类条件对齐不会（E36）─────────────────────────
def t23():
    e36 = J('E36_fpr_metrics')
    tb = Table(23, '表 23')
    tb.header = ['特征集 / 目标', 'AUC 无校正', 'AUC 整批对齐', 'AUC 类条件对齐',
                 'TPR@FPR=1% 无校正', 'TPR@FPR=1% 整批对齐', 'TPR@FPR=1% 类条件对齐',
                 '$\\rho$ 整批对齐', '$\\rho$ 类条件对齐']
    d_bn, d_cc = [], []
    for fs in ['15D', 'char_2_4']:
        for t in TARGETS:
            c = e36['feature_sets'][fs][t]['cells']
            bn, cc = c['bn'], c['bnc']
            d_bn.append(bn['TPR@FPR=0.01'] - c['none']['TPR@FPR=0.01'])
            d_cc.append(cc['TPR@FPR=0.01'] - c['none']['TPR@FPR=0.01'])
            tb.rows.append(['%s / %s' % (FS_LABEL[fs], t), f3(c['none']['AUC']),
                            f3(bn['AUC']), f3(cc['AUC']),
                            f3(c['none']['TPR@FPR=0.01']), f3(bn['TPR@FPR=0.01']),
                            f3(cc['TPR@FPR=0.01']),
                            f3(bn['spearman_vs_none']), f3(cc['spearman_vs_none'])])
    rhos_bn = [float(r[7]) for r in tb.rows]
    tb.note = ('整批对齐（BN）按每个维度各自重标定，等效于换了一个线性判别方向，'
               '因此会改变分数排序：%d 组配置上 $\\rho$ 全部小于 1（%.4f~%.4f），'
               'TPR@FPR=1%% 全部下降（平均 %+.3f，最大 %+.3f）。'
               '类条件对齐（BNC）在同一批配置上 $\\rho$ 严格为 1.000、'
               'AUC 与 TPR@FPR 完全不变（平均 %+.3f）。'
               '这是"必须做类条件对齐而非整批对齐"独立于残留漂移的第二条证据。'
               % (len(d_bn), min(rhos_bn), max(rhos_bn),
                  sum(d_bn) / len(d_bn), min(d_bn),
                  sum(d_cc) / len(d_cc)))
    return tb


# ── 表 25：与零样本检测器族的同条件对比（E38）───────────────────────────────
def t25():
    e38 = J('E38_zeroshot_baseline')
    tb = Table(25, '表 25')
    tb.header = ['目标 LLM', 'PPL', 'PPL$\\div$zlib', 'GLTR',
                 '本文 15D', '本文 char 2-4', '差值']
    tg = e38['targets']
    gaps = []
    for t in TARGETS:
        if t not in tg:
            continue
        r = tg[t]
        ours = max(r['ours_15D_tta']['AUC'], r['ours_char_2_4_tta']['AUC'])
        base = max(r['ppl']['AUC'], r['ppl_zlib']['AUC'], r['gltr']['AUC'])
        gaps.append(ours - base)
        tb.rows.append([t, f3(r['ppl']['AUC']), f3(r['ppl_zlib']['AUC']),
                        f3(r['gltr']['AUC']),
                        f3(r['ours_15D_tta']['AUC']),
                        '**%s**' % f3(r['ours_char_2_4_tta']['AUC']),
                        sg3(ours - base)])
    tb.note = ('全部为 AUC（阈值无关）。零样本基线用开源替代模型 %s 实现：'
               '目标域是闭源商用 LLM，部署方拿不到生成模型权重，只能另找模型算概率。'
               '文本截断到前 %d 个词元，每类 %d 条。GLTR 与本文方法同样只使用 ChatGPT '
               '源域训练，比较最为对等。差值 = 本文最优 − 基线最优，均值为 %+.3f。'
               '**该基线强度应视为下界**：DetectGPT/Binoculars 原论文使用 7B~70B 级模型，'
               '受算力限制本次只能用 0.5B 级替代模型（无 GPU 环境下 22 核 CPU 约 '
               '16 词元/秒），无法复现其规模。'
               % (e38['causal_model'], e38['max_len'], e38['n_per_class'],
                  sum(gaps) / len(gaps)))
    return tb


BUILDERS = {'8': t8, '9': t9, '10': t10, '11': t11, '12': t12, '13': t13, '14': t14,
            '15': t15, '17': t17, '18': t18, '21': t21,
            '26': t26, '27': t27, '28': t28, '19': t19, '20': t20,
            '22': t22, '24': t24, '16': t16, '23': t23, '25': t25}

# 随 --ubc 切换校正口径的表
UBC_TABLES = ('11', '13', '15', '22', '24')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    ap.add_argument('--ubc', default='oracle', choices=['em', 'auto', 'oracle'])
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()

    keys = [k for k in BUILDERS if (not args.only or k in args.only.split(','))]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k in sorted(keys, key=int):
        fn = BUILDERS[k]
        try:
            tb = fn(args.ubc) if k in UBC_TABLES else fn()
        except TypeError:
            tb = fn()
        p = out / ('table_%s.md' % k)
        with io.open(p, 'w', encoding='utf-8') as f:
            f.write(tb.render())
        print('wrote %s  (%d rows)' % (p.name, len(tb.rows)))


if __name__ == '__main__':
    main()
