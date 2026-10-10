# -*- coding: utf-8 -*-
"""
按依赖顺序批量执行实验。

  python run_all.py                        # 跑全部（缺依赖的自动跳过）
  python run_all.py --only E19,E26         # 只跑指定实验
  python run_all.py --skip E33,E32         # 跳过指定实验（例如耗时的生成类）

结果写入 results/<实验名>.json，日志同时打印到标准输出。
"""
import argparse
import glob
import importlib.util
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
DATA = os.path.join(ROOT, 'data')

# (脚本, 说明, 额外依赖, 是否耗时)
PLAN = [
    ('E17_crossllm_ghostbuster.py', '跨 LLM 特征对照（15D vs TF-IDF）', None, False),
    ('E18_indomain_ref.py', '同协议 in-domain 参照', None, False),
    ('E19_invariance_deficit.py', 'Ξ 与跨 LLM 衰减的关联', None, True),
    ('E20_clida_main.py', 'CLIDA 混合数据效率', None, True),
    ('E26_tta.py', 'TTA 主实验（批对齐 vs 类条件对齐）', None, True),
    ('E27_roberta_tta.py', '冻结 RoBERTa 上的 TTA', 'torch', True),
    ('E28_tta_lambda.py', 'TTA-λ 统一族与偏差-方差权衡', 'torch', True),
    ('E29_tta_robust.py', '鲁棒 E 步', 'torch', True),
    ('E30_tta_select.py', '无标签自动选参', 'torch', True),
    ('E31_batch_size.py', '批次规模敏感性（15D / char）', None, False),
    ('E32_roberta_batch.py', '批次规模敏感性（RoBERTa）', 'torch', True),
    ('E33_paraphrase_attack.py', '真实改写攻击下的鲁棒性', 'torch', True),
    ('E33b_paraphrase_diag.py', '改写攻击损失的机制分解', None, False),
    ('E34_train_size.py', '训练规模敏感性', None, False),
    ('E35_prior_scan.py', '批次类先验 π 失衡的边界', None, False),
    ('E36_fpr_metrics.py', '误报约束下的工作点（TPR@FPR / FPR@z=0 / ECE）', None, False),
    ('E37_supervised_ceiling.py', '目标域监督上界与恢复率', None, True),
    ('E38_zeroshot_baseline.py', '零样本检测器族同条件对比', 'transformers', True),
    ('E10_reproducibility.py', '可复现性打包', None, False),
]

NOISE = ('Prefix dict', 'Loading model', 'Dumping', 'Building prefix')


def has_module(name):
    return importlib.util.find_spec(name) is not None


def has_corpus():
    return (os.path.exists(os.path.join(DATA, 'all.jsonl'))
            and len(glob.glob(os.path.join(DATA, 'collected', '*.jsonl'))) > 0)


def run(script, args=None):
    print('\n' + '=' * 72)
    print('>>> %s' % script)
    print('=' * 72)
    t = time.time()
    r = subprocess.run([PY, os.path.join(ROOT, script)] + (args or []),
                       cwd=ROOT, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    for line in (r.stdout or '').splitlines():
        if not any(s in line for s in NOISE):
            print(line)
    if r.returncode != 0:
        print('[失败 rc=%d] %s' % (r.returncode, (r.stderr or '')[-500:]))
    print('--- %.1fs ---' % (time.time() - t))
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='', help='只跑这些实验，逗号分隔')
    ap.add_argument('--skip', default='', help='跳过这些实验，逗号分隔')
    ap.add_argument('--allow-slow', action='store_true', help='包含耗时实验')
    a = ap.parse_args()

    only = {s.strip() for s in a.only.split(',') if s.strip()}
    skip = {s.strip() for s in a.skip.split(',') if s.strip()}

    if not has_corpus():
        print('缺少语料：需要 data/all.jsonl 与 data/collected/*.jsonl')
        return 1

    ok, skipped = [], []
    for script, desc, dep, slow in PLAN:
        key = script.split('_')[0]
        if only and key not in only:
            continue
        if key in skip:
            skipped.append(key)
            continue
        if dep and not has_module(dep):
            print('\n[跳过] %s —— %s（缺少 %s）' % (key, desc, dep))
            skipped.append(key)
            continue
        if slow and not a.allow_slow and not only:
            print('\n[跳过] %s —— %s（耗时实验，加 --allow-slow 或 --only %s 运行）'
                  % (key, desc, key))
            skipped.append(key)
            continue
        (ok if run(script) else skipped).append(key)

    print('\n' + '=' * 72)
    print('完成 %d / 未执行 %d' % (len(ok), len(skipped)))
    if skipped:
        print('未执行：%s' % ', '.join(skipped))
    return 0


if __name__ == '__main__':
    sys.exit(main())
