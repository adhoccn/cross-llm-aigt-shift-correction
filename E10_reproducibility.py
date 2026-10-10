# -*- coding: utf-8 -*-
"""
可复现性打包：生成环境、种子、超参与结果索引的快照。

  dist/requirements.txt      精确版本锁定
  dist/ENVIRONMENT.md        Python / 库 / OS 快照
  dist/SEEDS.md              随机种子的使用位置
  dist/HYPERPARAMS.md        XGBoost 超参与搜索范围
  dist/RESULTS_INDEX.json    全部实验结果的索引与校验和
"""
import sys, os, json, io, platform, subprocess, time, hashlib, glob
ROOT = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(ROOT, 'dist'); RES = os.path.join(ROOT, 'results')
os.makedirs(DIST, exist_ok=True)


def main():
    import numpy, sklearn, scipy
    try:
        import xgboost; xgbv = xgboost.__version__
    except Exception:
        xgbv = 'not-installed'
    try:
        import jieba; jbv = jieba.__version__
    except Exception:
        jbv = 'not-installed'

    src_req = os.path.join(ROOT, 'requirements.txt')
    if os.path.exists(src_req):
        req = [l.strip() for l in io.open(src_req, encoding='utf-8')
               if l.strip() and not l.startswith('#')]
    else:
        req = ['numpy==%s' % numpy.__version__, 'scikit-learn==%s' % sklearn.__version__,
               'scipy==%s' % scipy.__version__, 'xgboost==%s' % xgbv, 'jieba==%s' % jbv]
    with io.open(os.path.join(DIST, 'requirements.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(req) + '\n')

    pkgs = {k: v for k, v in (x.split('==', 1) for x in req if '==' in x)}
    env = {'python': platform.python_version(), 'os': platform.platform(),
           'machine': platform.machine(), 'packages': pkgs,
           'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    with io.open(os.path.join(DIST, 'ENVIRONMENT.md'), 'w', encoding='utf-8') as f:
        f.write('# 运行环境快照\n\n')
        f.write('- Python %s / %s / %s\n' % (env['python'], env['os'], env['machine']))
        for k, v in env['packages'].items():
            f.write('- %s == %s\n' % (k, v))

    seeds = """# 随机种子使用位置（全部固定为 42）
| 位置 | 用途 |
|---|---|
| core/dataio.py pair_stratified_kfold(seed=42) | 折划分 |
| core/evalkit.py make_model(seed=42) | XGBoost random_state |
| E0/E4/E6/E9 主脚本 rng = default_rng(42) | 抽样 |
| core/statskit.py bootstrap_diff(seed=42) | bootstrap 重采样 |
| E1_mix_ratio.py | 混合比例抽样的 seed=42+fold_j |
"""
    io.open(os.path.join(DIST, 'SEEDS.md'), 'w', encoding='utf-8').write(seeds)

    hyper = """# XGBoost 超参（core/evalkit.py make_model）
n_estimators = 200
max_depth = 4
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8
reg_lambda = 1.0
random_state = 42
eval_metric = logloss
搜索范围：本文未做系统化的超参搜索，故不提供搜索区间；上述取值为固定设置。
"""
    io.open(os.path.join(DIST, 'HYPERPARAMS.md'), 'w', encoding='utf-8').write(hyper)

    idx = {}
    for p in sorted(glob.glob(os.path.join(RES, '*.json'))):
        with io.open(p, 'rb') as f:
            b = f.read()
        idx[os.path.basename(p)] = {'bytes': len(b),
                                    'sha256': hashlib.sha256(b).hexdigest()[:16]}
    with io.open(os.path.join(DIST, 'RESULTS_INDEX.json'), 'w', encoding='utf-8') as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)
    print('可复现性打包完成 → dist/')
    for k, v in idx.items():
        print('  %-28s %8d bytes  sha256:%s' % (k, v['bytes'], v['sha256']))


if __name__ == '__main__':
    main()
