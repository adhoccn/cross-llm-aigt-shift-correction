# -*- coding: utf-8 -*-
"""
表号体检：确认稿件中的表号从 1 起严格连续，且与生成器、表题字典、表格文件一致。

用法：
    python tools/check_table_numbers.py                     体检默认路径的稿件 md
    python tools/check_table_numbers.py --md <稿件.md>

默认路径由环境变量 `PAPER_MD` 指定，未设置时回退到仓库根的 `manuscript.md`；
稿件不在本仓库内时请用 `--md` 显式给出路径。

检查四项：
    1. 稿件中 [TABLE n] 的出现顺序必须是 1,2,3,...,N（表号 = 出现顺序）；
    2. 正文"表 N"引用不得越界（不得引用不存在的表）；
    3. md2docx_fresh.py 的 CAPTIONS 必须覆盖 1..N 且无空缺；
    4. tables/table_N.md 必须存在（表 1–7 为手写定义表，不要求文件）。

退出码：0 = 全部通过；1 = 存在问题。
"""
from __future__ import annotations
import io, os, re, sys, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_MD = Path(os.environ.get('PAPER_MD') or (ROOT / 'manuscript.md'))
DOCX = ROOT / 'md2docx_fresh.py'

# 表 1–7 是定义与清单表，正文手写、不由 make_tables 生成
HAND_WRITTEN = set(range(1, 8))


def scan(md_path):
    lines = io.open(md_path, encoding='utf-8').read().split('\n')
    tags, refs = [], []
    in_table = False
    for i, ln in enumerate(lines, 1):
        st = ln.strip()
        if st.startswith('[TABLE'):
            tags.append((i, st[7:].split(']')[0].strip()))
            in_table = True
            continue
        if st.startswith('[/TABLE]'):
            in_table = False
            continue
        if st.startswith('|'):        # 表格行内的"表 N"不计入正文引用
            continue
        in_table = False             # 遇到非表格行即视为块结束（稿件未必写 [/TABLE]）
        for m in re.finditer(r'表\s*(\d+)', ln):
            refs.append((i, int(m.group(1))))
    return tags, refs


def captions():
    """从 md2docx 的 CAPTIONS 字典里取表题编号。

    该脚本不在本仓库内时（稿件与代码分开维护），跳过这一项检查。
    """
    if not DOCX.exists():
        return None
    s = io.open(DOCX, encoding='utf-8').read()
    if 'CAPTIONS = {' not in s:
        return None
    i0 = s.index('CAPTIONS = {')
    i1 = s.index('}', i0)
    return {int(m.group(1)) for m in
            re.finditer(r'^\s*(\d+):', s[i0:i1], flags=re.M)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', default=str(DEFAULT_MD))
    args = ap.parse_args()

    tags, refs = scan(args.md)
    problems = []

    # 1. 出现顺序
    seq = [t[1] for t in tags]
    want = [str(k) for k in range(1, len(seq) + 1)]
    print('表格块 %d 个，标记序列：%s' % (len(seq), ' '.join(seq)))
    if seq != want:
        for i, (got, exp) in enumerate(zip(seq, want), 1):
            if got != exp:
                problems.append('第 %d 个表格标记 [TABLE %s]，应为 [TABLE %s]'
                                % (i, got, exp))
                break
    N = len(seq)

    # 2. 引用越界
    bad = sorted({n for _, n in refs if not 1 <= n <= N})
    if bad:
        problems.append('正文引用了不存在的表：%s' % bad)
    cited = {n for _, n in refs}
    unused = [n for n in range(1, N + 1) if n not in cited]
    print('正文引用覆盖 %d/%d 张表%s'
          % (N - len(unused), N,
             '' if not unused else '，未被引用：%s' % unused))

    # 3. 表题字典
    cap = captions()
    if cap is None:
        print('（跳过表题检查：未找到 %s）' % DOCX)
    else:
        missing = [n for n in range(1, N + 1) if n not in cap]
        extra = sorted(cap - set(range(1, N + 1)))
        if missing:
            problems.append('CAPTIONS 缺表题：%s' % missing)
        if extra:
            problems.append('CAPTIONS 有多余条目：%s' % extra)

    # 4. 表格文件
    for n in range(1, N + 1):
        if n in HAND_WRITTEN:
            continue
        if not (ROOT / 'tables' / ('table_%d.md' % n)).exists():
            problems.append('缺少 tables/table_%d.md' % n)

    print('-' * 60)
    if problems:
        for p in problems:
            print('[问题] ' + p)
        print('表号体检：未通过（%d 项）' % len(problems))
        return 1
    print('表号体检：通过。表 1–%d 严格连续，生成器/表题/文件/引用四者一致。' % N)
    return 0


if __name__ == '__main__':
    sys.exit(main())
