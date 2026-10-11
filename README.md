# Cross-LLM Chinese AIGT Detection: Discriminative Shift and Unsupervised Batch Correction

> **Reproduction code for cross-LLM Chinese AI-generated-text detection: predictable
> discriminative shift and unsupervised batch correction.**

Reproduction code for the paper *"面向未知大语言模型的中文 AI 生成文本检测：判别漂移的可预测性与无标签批校正"*
(*Detecting Chinese AI-Generated Text from Unknown LLMs: Predictability of Discriminative Shift
and Unsupervised Batch Correction*). Author identity is withheld in this repository for
double-blind review. A Chinese version of this document is in `README_CN.md`.

**Live demo of the deployed service (TTAScope):** <http://106.54.192.87:8501/tta/> — submit a
batch of Chinese text to see per-item verdicts and the drift diagnostic. No data is retained
server-side.

## What the paper claims, in one page

A detector trained on one source LLM (ChatGPT) degrades badly on text from an unseen LLM.
This repository implements and measures two things:

1. **Discriminative shift** $\Xi_{\mathrm{eff}}$ — the component of the target-domain mean shift
   that lies along the source discriminative direction. Unlike raw distribution shift, it
   *predicts* the F1 drop (30 configurations, mean absolute error 0.042) and supports an
   analytic form $\mathrm{F1} \approx 1 - \Phi(-\Delta'/2)$ with $\Delta' = \Delta - \Xi_{\mathrm{eff}}$.
2. **Unsupervised batch correction (UBC)** — at deployment time, given a *batch* of unlabeled
   target texts, class-conditional alignment moves the two class means back onto the source
   geometry using soft posteriors (an EM-style E step). It needs **no target labels and no
   access to the target model**.

Two properties are worth checking first if you only read one table:

- Class-conditional alignment is a **strictly monotone recalibration** of the score:
  Spearman $\rho = 1.000$ and $\Delta\mathrm{AUC} \le 0.0005$ on all 10 configurations.
  The gain is entirely at the decision threshold, not in ranking quality. This is a
  *characterisation*, not a defect — it is why the method is judged at fixed $z=0$ and why
  fixed-FPR deployments gain nothing (Tables 22–24).
- It **fails detectably**: when the batch class prior $\pi \to 1$ (all-AI batch), the class
  balance prior does *not* trigger and correction actively hurts (macro-F1 $-0.107$).
  See `E35_prior_scan.py` and Tables 19–20.

## Repository layout

```
├── core/               Shared library: features, evaluation, statistics, geometry
│   ├── features.py              TF-IDF / character n-gram / 15-D handcrafted features
│   ├── feature_extractor_15d.py Dependency-free 15-D stylometric features
│   ├── evalkit.py               5-fold x 5-seed protocol, macro-F1, significance tests
│   ├── statskit.py              Shift, drift and separation estimators
│   ├── invariance.py            Environment scatter S_E vs between-class scatter S_B
│   └── dataio.py                Corpus loading
├── E*.py               One script per experiment (see the mapping below)
├── tools/
│   ├── make_tables.py           results/*.json -> tables/*.md  (single source of numbers)
│   ├── check_tables.py          Re-derive every number in the manuscript and diff
│   ├── check_table_numbers.py   Table numbering sanity check
│   ├── make_figs.py             Figures (PDF vector + 600 dpi PNG)
│   ├── dump_results.py          Result summary
│   └── se_sb_angle.py           S_E / S_B subspace angle
├── gen_data.py         Corpus collection from OpenAI-compatible endpoints
├── run_all.py          Batch entry point
├── config.yaml         Collection endpoint template (keys from environment)
├── results/            All raw JSON results (numbers in the paper come from here)
└── tables/             Generated manuscript tables (Tables 8-28)
```

Every number in the manuscript is produced by `tools/make_tables.py` from `results/*.json`;
nothing is transcribed by hand.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Core experiments need only `numpy / scipy / scikit-learn / xgboost / jieba / statsmodels`.
`torch` and `transformers` are required only for the deep-representation experiments
(E27, E28, E29, E30, E32, E33, E38) — see `requirements-deep.txt`; `run_all.py` skips those
automatically when they are absent. All experiments run on CPU.

## Data

The corpus is **not** included (size and licensing). Two parts:

- **Human-written text**: HC3-Chinese (`Hello-SimpleAI/HC3`, `zh` split), publicly available.
- **AI-generated text**: 1 000 question-answer pairs per target LLM, collected with
  `gen_data.py` against OpenAI-compatible endpoints. Fill in `config.yaml`, export the
  API keys as environment variables, then:

```bash
python gen_data.py --llm deepseek --n 1000 --seed 42
#  -> data/collected/deepseek_seed42_n1000.jsonl  (override with --out)
```

`gen_data.py` resumes from what is already on disk, so an interrupted run can be repeated.
Expected layout under `data/`:

```
data/all.jsonl                          # HC3-Chinese, six domains merged
data/<domain>.jsonl                     #   ... and per domain (baike, law, psychology,
                                        #       open_qa, medicine, finance)
data/collected/<llm>_seed42_n1000.jsonl # collected LLM answers (archive under this name)
data/para_attack_corpus.jsonl           # paraphrase-attack corpus (E33)
```

HC3 records carry `source / question / human_answers[] / chatgpt_answers[]`; the collected
records carry `question / answer / domain` plus `ts / prompt_tokens / completion_tokens /
latency_ms`. Question-level isolation is enforced when splitting (same question never spans
folds), which matters: violating it inflates every cross-LLM number.

Set `TTA_DATA_DIR` to override the data root.

## Reproducing

```bash
python run_all.py                    # default: slow experiments skipped
python run_all.py --only E26,E30     # a subset (comma separated)
python run_all.py --allow-slow       # include the slow ones as well

python tools/make_tables.py --ubc oracle     # regenerate tables/ from results/
python tools/check_tables.py --md <manuscript.md> --ubc oracle
python tools/check_table_numbers.py --md <manuscript.md>
```

`check_tables.py` re-derives every table from the raw JSON and reports any disagreement
(exit code 1 on mismatch), so it doubles as a minimal CI assertion.
`--ubc oracle` selects the oracle variant column and matches the manuscript; replace with
`auto` for the fully label-free operating point.

The two checking tools look for the manuscript Markdown at `$PAPER_MD`
(default `manuscript.md` in the repository root); pass `--md` explicitly if it lives
elsewhere. The manuscript itself is not part of this repository, so `--md` is normally
required; leave `manuscript.md` out of version control if you keep it here.

### Experiment → table mapping

| Script | What it establishes | Tables |
|---|---|---|
| E17 | 15-D vs TF-IDF baselines under one protocol | 13 |
| E18 | Same-protocol in-domain reference | 2, 13 |
| E19 | $\Xi_{\mathrm{eff}}$ definition, analytic F1 prediction, granularity law, Corollary 3 | 2–7 |
| E20 | CLIDA, per-dimension modulation, $S_E$/$S_B$ overlap | 26–28 |
| E26 | Batch vs class-conditional alignment, gain vs initial drift, stacking with mixing | 11–14 |
| E27 | Frozen RoBERTa representations | 11 |
| E28 | Correction strength $\lambda$ | 17 |
| E29 | E-step variants (trim / conf / oracle) | 9, 15 |
| E30 | One-factor vs two-factor error model; label-free variant selection | 8, 10 |
| E31 / E32 | Batch-size sensitivity (15-D / char / RoBERTa) | 18 |
| E33 / E33b | Real paraphrase attack; repairable vs unrepairable loss | 21 |
| E34 | Training-set size sensitivity | §3.3 |
| E35 | Batch class-prior imbalance $\pi$ | 19–20 |
| E36 | Fixed-FPR operating points, FPR@$z$=0, ECE, Spearman $\rho$ | 22–24 |
| E37 | Target-domain supervised ceiling and recovery rate | 16 |
| E38 | Zero-shot detector family (PPL / PPL÷zlib / GLTR) under the same protocol | 25 |
| E10 | Environment, seed and hyper-parameter snapshot | reproducibility |

## Notes and known constraints

- **E38 surrogate model.** The zero-shot baselines require an open-source surrogate
  (default `Qwen2.5-0.5B-Instruct`, ~1 GB, downloaded on first use). The original
  DetectGPT / Binoculars results use 7B–70B models; our CPU-only setup (≈16 tokens/s at
  0.5B) cannot reproduce that scale, so **Table 25 is a lower bound** on that family.
- **Compute.** Everything runs on CPU. The full suite takes a few hours; E33 (neural
  paraphrasing) and E38 (per-token scoring) dominate.
- **Table 18** in the manuscript lists $n \ge 50$ only; the generator keeps the full grid
  from $n=10$ because gains turn negative below the usable batch size.
- **Randomness.** 5-fold × 5 seeds (42, 123, 456, 789, 2024). Fold partitions are shared
  across configurations, so seed-aggregated paired tests are also reported where relevant.

## License

MIT — see `LICENSE`. The corpus is not distributed here; check the license of each source
dataset and model before use.
