# -*- coding: utf-8 -*-
"""
稿件数字体检：把 paper 中的表格数字与 results/*.json 的重算值逐格比对。

用法：
    python tools/check_tables.py                       体检默认路径的稿件 md
    python tools/check_tables.py --md <稿件.md> --tol 0.0015
    python tools/check_tables.py --fix-hint            额外输出"最接近的口径"提示

默认路径由环境变量 `PAPER_MD` 指定，未设置时回退到仓库根的 `manuscript.md`；
稿件不在本仓库内时请用 `--md` 显式给出路径。

退出码：0 = 全部一致；1 = 存在超出容差的差异（可作为最小 CI 的断言）。
"""
from __future__ import annotations
import io, os, sys, re, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import make_tables as MT  # noqa: E402

NUM = re.compile(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?')
DEFAULT_MD = Path(os.environ.get('PAPER_MD') or (ROOT / 'manuscript.md'))
REPORT = ROOT / 'tables' / 'AUDIT.md'


def norm_cell(s):
    return s.replace('\u2212', '-').replace('−', '-').replace('–', '-')


def nums(s):
    return [float(x) for x in NUM.findall(norm_cell(s))]


def parse_md_tables(md_path):
    """返回 {表号: [[单元格...], ...]}（只含数据行）。"""
    lines = io.open(md_path, encoding='utf-8').read().split('\n')
    out, cur, rows = {}, None, []
    for ln in lines:
        st = ln.strip()
        if st.startswith('[TABLE'):
            cur = st[len('[TABLE'):].split(']')[0].strip()
            rows = []
            continue
        if cur is None:
            continue
        if st.startswith('|'):
            cells = [c.strip() for c in st.strip('|').split('|')]
            if set(''.join(cells)) <= set('-: '):
                continue
            rows.append(cells)
            out[cur] = rows
        elif st == '' or st.startswith('*') or st.startswith('!'):
            continue
        else:
            if cur in out:
                out[cur] = rows
            cur = None
    # 去掉表头行
    return {k: v[1:] for k, v in out.items() if len(v) > 1}


SKIP_LAST = {'12': 1}   # 表 12 末列是文字（代表场景），不参与数值比对


def compare(md_rows, gen_rows, tol, skip_last=0):
    """逐格比对，返回差异列表 [(行号, 列号, 稿件值, 复现值)]。"""
    diffs = []
    if skip_last:
        md_rows = [r[:-skip_last] for r in md_rows]
        gen_rows = [r[:-skip_last] for r in gen_rows]
    for i, (a, b) in enumerate(zip(md_rows, gen_rows)):
        na, nb = nums(' '.join(a)), nums(' '.join(b))
        if len(na) != len(nb):
            diffs.append((i, -1, '数字个数 %d' % len(na), '数字个数 %d' % len(nb)))
            continue
        for j, (x, y) in enumerate(zip(na, nb)):
            scale = max(1.0, abs(x))
            if abs(x - y) / scale > tol and abs(x - y) > tol:
                diffs.append((i, j, x, y))
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', default=str(DEFAULT_MD))
    ap.add_argument('--tol', type=float, default=0.0012)
    ap.add_argument('--out', default=str(REPORT))
    ap.add_argument('--fix-hint', action='store_true')
    ap.add_argument('--ubc', default='oracle', choices=['em', 'auto', 'oracle'],
                    help='表 %s 的校正口径，决定"复现值"取哪一列'
                         % '/'.join(MT.UBC_TABLES))
    args = ap.parse_args()

    md = parse_md_tables(args.md)
    gen = {}
    variants = {}
    for k, fn in MT.BUILDERS.items():
        if k in MT.UBC_TABLES:
            for mode in ('em', 'auto', 'oracle'):
                tb = fn(mode)
                variants[(k, mode)] = tb
                if mode == args.ubc:
                    gen[k] = tb
        else:
            gen[k] = fn()

    lines = ['# 稿件数字体检报告', '',
             '稿件：%s' % os.path.relpath(args.md, ROOT.parent),
             '容差：%g（相对或绝对）' % args.tol,
             '校正口径：%s' % args.ubc,
             '结果来源：results/*.json，由 tools/make_tables.py 生成',
             '',
             '差异即"稿件数字 → 复现数字"的替换清单。', '']
    total_diff = 0
    for k in sorted(gen, key=int):
        if k not in md:
            lines.append('## 表 %s：稿件中未找到，跳过' % k)
            continue
        d = compare(md[k], gen[k].rows, args.tol, SKIP_LAST.get(k, 0))
        head = '## 表 %s：%s' % (k, '一致' if not d else '存在 %d 处差异' % len(d))
        lines.append(head)
        if d:
            total_diff += len(d)
            lines.append('')
            lines.append('| 行 | 列 | 稿件值 | 复现值 | 差 |')
            lines.append('|---|---|---|---|---|')
            for i, j, x, y in d[:40]:
                label = md[k][i][0] if i < len(md[k]) else ''
                if isinstance(x, str) or isinstance(y, str):
                    lines.append('| %s | - | %s | %s | - |' % (label, x, y))
                else:
                    lines.append('| %s | %d | %.4g | %.4g | %+.4g |' % (label, j, x, y, y - x))
        if args.fix_hint and (k, 'em') in variants:
            scores = {}
            for mode in ('em', 'auto', 'oracle'):
                dd = compare(md[k], variants[(k, mode)].rows, args.tol, SKIP_LAST.get(k, 0))
                scores[mode] = len(dd)
            best = min(scores, key=lambda m: scores[m])
            lines.append('')
            lines.append('口径匹配（差异格数）：' + '、'.join(
                '%s=%d' % (m, scores[m]) for m in ('em', 'auto', 'oracle')) +
                ' → 最接近：**%s**' % best)
        lines.append('')

    if total_diff == 0:
        lines.append('全部受检表格的数字与 results/*.json 一致。')
    else:
        lines.append('共 %d 处差异。处理方式二选一：改稿件数字，或改生成口径并重跑实验。'
                     % total_diff)
        lines.extend(['', '## 替换清单在稿件中的出现位置', ''])
        raw = io.open(args.md, encoding='utf-8').read().split('\n')
        seen = set()
        for k in sorted(gen, key=int):
            if k not in md:
                continue
            for i, j, x, y in compare(md[k], gen[k].rows, args.tol, SKIP_LAST.get(k, 0)):
                if isinstance(x, str):
                    continue
                key = (k, i, j)
                if key in seen:
                    continue
                seen.add(key)
                pats = ['%.3f' % x, '%+.3f' % x, '%.3f' % abs(x), '%.2f' % x]
                kw = ('UBC', 'EM', '校正', '增益', '混合', 'TTA', '表 1')
                hits = []
                for n, ln in enumerate(raw, 1):
                    if ln.lstrip().startswith('|'):
                        continue
                    if any(p in ln for p in pats) and any(w in ln for w in kw):
                        hits.append(n)
                where = ('正文第 ' + '、'.join(str(h) for h in hits[:8]) + ' 行'
                         if hits else '仅出现在表格内')
                lines.append('- 表 %s 第 %d 行：`%s` → `%s`（%s）' %
                             (k, i + 1, '%.3f' % x, '%.3f' % y, where))
    txt = '\n'.join(lines)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    io.open(args.out, 'w', encoding='utf-8').write(txt)
    print(txt[:4000])
    print('\n[report] %s' % args.out)
    return 1 if total_diff else 0


if __name__ == '__main__':
    sys.exit(main())
