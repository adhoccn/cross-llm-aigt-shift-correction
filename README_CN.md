# 跨 LLM 中文 AI 生成文本检测：判别漂移与无标签批校正（复现代码）

本仓库是论文《面向未知大语言模型的中文 AI 生成文本检测：判别漂移的可预测性与无标签批校正》的
复现代码。为配合双盲评审，仓库中不出现作者姓名与单位。英文说明见 `README.md`。

## 方法要点

1. **判别漂移 $\Xi_{\mathrm{eff}}$**：目标域均值漂移在源域判别方向上的投影。与"原始分布漂移"不同，
   它能**预测**精度衰减（30 组配置，平均绝对误差 0.042），并给出解析式
   $\mathrm{F1}\approx 1-\Phi(-\Delta'/2)$，$\Delta'=\Delta-\Xi_{\mathrm{eff}}$。
2. **无标签批校正（UBC）**：部署时收到**一批**无标签目标文本，用分类器软后验做 E 步、估计目标域
   类条件均值，再做类条件对齐。不需要目标域标签，也不需要访问目标模型。

只想看一组表的话，建议先看表 S4–S6：类条件对齐是分数空间的**严格单调重标定**
（斯皮尔曼 $\rho=1.000$，$\Delta\mathrm{AUC}\le 0.0005$），收益全部落在决策阈值位置，
不改变排序质量。这是方法性质的刻画，不是缺陷；也正因如此，按固定误报预算运营时本方法不带来
检出率提升。另一面是它**失效可检测**：批次类先验 $\pi\to1$（整批同源）时类平衡先验不会触发，
校正反而造成 $-0.107$ 的损失（见 `E35_prior_scan.py`、表 S1–S2）；可用的前置补救是 $r_{AI}>0.64$ 时拒绝校正（`E39_pi_estimator.py`、表 S11）。

## 目录结构

```
core/        公共库：特征、评估协议、统计量、几何
E*.py        每个实验一个脚本（映射表见 README.md）
tools/       make_tables（JSON → 表格，数字唯一来源）、check_tables（逐格复算体检）、
             check_table_numbers（表号体检）、make_figs、dump_results、se_sb_angle
gen_data.py  语料采集（OpenAI 兼容接口）
run_all.py   批量入口
results/     全部原始 JSON 结果
tables/      由 results 生成的稿件表格（编号沿用旧稿，映射见英文 README）
```

稿件中的数字一律由 `tools/make_tables.py` 从 `results/*.json` 生成，不手抄。

## 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

核心实验只需 `numpy / scipy / scikit-learn / xgboost / jieba / statsmodels`；
`torch` 与 `transformers` 仅深度表征实验（E27/E28/E29/E30/E32/E33/E38）需要，见
`requirements-deep.txt`；缺失时 `run_all.py` 会自动跳过。全部实验可在 CPU 上运行。

## 数据

语料不随仓库分发。两部分：

- **人类文本**：HC3-Chinese（`Hello-SimpleAI/HC3` 的 `zh` 子集），公开数据集。
- **AI 文本**：每个目标 LLM 采集 1000 组问答，用 `gen_data.py` 调 OpenAI 兼容接口生成。
  填写 `config.yaml`、把 API key 写入环境变量后运行：

```bash
python gen_data.py --llm deepseek --n 1000 --seed 42
#  -> data/collected/deepseek_seed42_n1000.jsonl（可用 --out 覆盖）
```

`gen_data.py` 落盘为 jsonl 且已存在的 question 会跳过，中断后可重复运行续采。
实验读取的目录结构如下：

```
data/all.jsonl                          # HC3-Chinese，六域合并
data/<domain>.jsonl                     #   另有分域文件（baike / law / psychology /
                                        #   open_qa / medicine / finance）
data/collected/<llm>_seed42_n1000.jsonl # 自建多 LLM 语料（采集后按此命名归档）
data/para_attack_corpus.jsonl           # 改写攻击语料（E33）
```

HC3 每行含 `source / question / human_answers[] / chatgpt_answers[]`；自建语料每行含
`question / answer / domain`，另有 `ts / prompt_tokens / completion_tokens / latency_ms`。
划分时强制**问题级隔离**（同一 question 衍生的文本不跨折）——这一点很关键，
违反会让所有跨 LLM 数字虚高。

## 复现

```bash
python run_all.py                 # 默认跳过耗时实验
python run_all.py --only E26,E30  # 只跑指定实验（逗号分隔）
python run_all.py --allow-slow    # 连耗时实验一起跑

python tools/make_tables.py --ubc oracle        # 由 results 重新生成表格
python tools/check_tables.py --md <稿件.md> --ubc oracle
python tools/check_table_numbers.py --md <稿件.md>
```

`check_tables.py` 会用原始 JSON 重算稿件里每一格并报告差异（不一致时退出码 1），可直接作为
CI 断言。`--ubc oracle` 为稿件口径；改 `auto` 即完全无标签的部署口径。
两个检查工具的默认稿件路径取自环境变量 `PAPER_MD`（默认仓库根的 `manuscript.md`），
也可用 `--md` 显式指定。稿件本身不在本仓库内，因此通常需要显式给 `--md`；
若把稿件放在仓库根，请用 `.gitignore` 排除。

## 已知约束

- **E38 替代模型**：零样本基线需要开源替代模型（默认 `Qwen2.5-0.5B-Instruct`，约 1 GB，首次运行下载）。
  DetectGPT / Binoculars 原论文使用 7B–70B 模型，本次 CPU 环境（0.5B 约 16 词元/秒）无法复现该规模，
  因此**表 S7 的基线强度应视为下界**。
- **算力**：全量跑完约数小时，E33（神经改写）与 E38（逐词元打分）占主要开销。
- **表 S22**：稿件只列 $n\ge50$ 的列（低于该规模增益转负），生成器保留从 $n=10$ 起的完整网格。
- **随机性**：5-fold × 5 seeds（42, 123, 456, 789, 2024）。

## 许可

MIT，见 `LICENSE`。语料不在本仓库分发，使用前请确认各数据集与模型的许可。
