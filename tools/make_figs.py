# -*- coding: utf-8 -*-
"""
生成投稿用图（读取 results/*.json）。
字体规范：字母/数字/希腊字母 = Times New Roman，中文 = 宋体（SimSun）。
实现方式：font.family fallback ['Times New Roman','SimSun']；
标签内不使用 mathtext（避免中文被 mathtext default 字体吞掉），
希腊字母与数学符号一律用 Unicode（Times 自带字形）。
输出：figs/figN_*.pdf（矢量）+ figs/figN_*.png（600dpi）。
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------- 全局字体与样式 ----------
rcParams["font.family"] = ["Times New Roman", "SimSun"]
rcParams["axes.unicode_minus"] = True
rcParams["axes.linewidth"] = 0.7
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"
rcParams["xtick.major.size"] = 3
rcParams["ytick.major.size"] = 3
rcParams["legend.frameon"] = False
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
OUT = os.path.normpath(os.path.join(HERE, "..", "figs"))
os.makedirs(OUT, exist_ok=True)

def J(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

FS_COLOR = {"15D": "#4C72B0", "word_1_1": "#DD8452", "word_1_2": "#55A868",
            "word_1_3": "#C44E52", "char_2_4": "#8172B3", "char_3_5": "#937860"}
FS_LABEL = {"15D": "15 维", "word_1_1": "词 1-gram", "word_1_2": "词 1-2gram",
            "word_1_3": "词 1-3gram", "char_2_4": "字符 2-4gram", "char_3_5": "字符 3-5gram"}
TGT_MARKER = {"deepseek": "o", "qwen": "s", "glm": "^", "gpt-4o": "D", "llama3": "v"}

def norm_fs(fs):
    return {"15D": "15D", "word_1_1": "word_1_1", "word_1_2": "word_1_2",
            "word_1_3": "word_1_3", "char_2_4": "char_2_4", "char_3_5": "char_3_5"}.get(fs, fs)

# ============================================================
# 图 1  判别漂移 vs 跨 LLM 精度衰减（E19，30 组）
# ============================================================
def fig1():
    d = J("E19_invariance_deficit.json")
    rows = d["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, key, tag, title in [
            (axes[0], "Xi_eff", "判别漂移 Ξ_eff", "(a) 判别方向投影漂移"),
            (axes[1], "Xi_raw", "全空间漂移 Ξ_raw", "(b) 全空间漂移（对照）")]:
        seen = set()
        for r in rows:
            fs = norm_fs(r["fs"])
            lab = FS_LABEL[fs] if fs not in seen else None
            seen.add(fs)
            ax.scatter(r[key], r["dF1"], s=26, marker=TGT_MARKER[r["tgt"]],
                       color=FS_COLOR[fs], label=lab, linewidths=0.4,
                       edgecolors="white", zorder=3)
        x = np.array([r[key] for r in rows]); y = np.array([r["dF1"] for r in rows])
        k, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, k * xs + b, color="#333333", lw=1.0, ls="--", zorder=2)
        rr = d["regression"]["Xi_eff_vs_DeltaF1" if key == "Xi_eff" else "Xi_raw_vs_DeltaF1"]
        ax.text(0.04, 0.93, f"r = {rr['pearson_r']:.3f},  R² = {rr['ols_r2']:.3f},  p = {rr['pearson_p']:.1e}",
                transform=ax.transAxes, fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", lw=0.6))
        ax.set_xlabel(f"{tag}（源域 → 目标 LLM）")
        ax.set_ylabel("跨 LLM 精度衰减 ΔF1")
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8.5)
    axes[1].legend(fontsize=7, loc="upper right", handletextpad=0.2,
                   borderaxespad=0.3, ncol=1)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig2_xi_scatter")

# ============================================================
# 图 2  推论 1 解析预测 + 粒度律（E19）
# ============================================================
def fig2():
    d = J("E19_invariance_deficit.json")
    rows = d["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    # (a) 解析预测 vs 实测
    ax = axes[0]
    seen = set()
    for r in rows:
        fs = norm_fs(r["fs"])
        lab = FS_LABEL[fs] if fs not in seen else None
        seen.add(fs)
        ax.scatter(r["F1_pred"], r["F1"], s=26, marker=TGT_MARKER[r["tgt"]],
                   color=FS_COLOR[fs], label=lab, linewidths=0.4,
                   edgecolors="white", zorder=3)
    lim = [0.70, 1.0]
    ax.plot(lim, lim, color="#333333", lw=1.0, ls="--", zorder=2)
    mae = d["analytic_prediction_error"]["MAE"]
    ax.text(0.04, 0.93, f"MAE = {mae:.3f}", transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", lw=0.6))
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("解析预测 1 − Φ(−Δ′/2)")
    ax.set_ylabel("实测 macro-F1")
    ax.set_title("(a) 推论 1：目标域精度解析预测", fontsize=9)
    # (b) 粒度律
    ax = axes[1]
    agg = {}
    for r in rows:
        agg.setdefault((norm_fs(r["fs"]), r["gran"]), []).append(r)
    xs, ys, es, cs = [], [], [], []
    for (fs, gran), rs in sorted(agg.items(), key=lambda kv: kv[0][1]):
        xs.append(gran); ys.append(np.mean([r["Xi_eff"] for r in rs]))
        es.append(np.std([r["Xi_eff"] for r in rs])); cs.append(FS_COLOR[fs])
    ax.bar(xs, ys, yerr=es, width=0.55, color=cs, ecolor="#555555",
           error_kw=dict(lw=0.8, capsize=2.5), zorder=3)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.10, f"{y:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["15 维", "词\n1-gram", "词\n1-2gram", "词\n1-3gram",
                        "字符\n2-4gram", "字符\n3-5gram"], fontsize=7.5)
    ax.set_xlabel("特征粒度（n-gram 阶数）")
    ax.set_ylabel("平均 Ξ_eff（5 个目标 LLM）")
    ax.set_title("(b) 推论 2：粒度律", fontsize=9)
    ax.tick_params(labelsize=8.5)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig1_analytic_granularity")

# ============================================================
# 图 3  推论 3：BN 残留闭式解（合成验证 + 真实数据）
# ============================================================
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    # (a) 合成验证
    ax = axes[0]
    synth = [
        (1.0, 0.40, 15, 0.3567, 0.3567), (1.0, 0.40, 50, 0.3573, 0.3573),
        (2.0, 0.80, 15, 0.5448, 0.5448), (2.0, 0.80, 50, 0.5386, 0.5386),
        (3.0, 1.20, 15, 0.5890, 0.5890), (3.0, 1.20, 50, 0.5802, 0.5802),
        (5.0, 2.00, 15, 0.5213, 0.5213), (5.0, 2.00, 50, 0.5112, 0.5112),
        (5.0, 2.00, 200, 0.5302, 0.5303), (4.0, 3.20, 50, 2.3307, 2.3306),
        (6.0, 1.20, 50, 0.1556, 0.1556)]
    obs = [s[3] for s in synth]; pred = [s[4] for s in synth]
    ax.scatter(pred, obs, s=30, color="#4C72B0", zorder=3, linewidths=0.4,
               edgecolors="white")
    lim = [-0.1, 2.6]
    ax.plot(lim, lim, color="#333333", lw=1.0, ls="--", zorder=2)
    dev = max(abs(o - p) for o, p in zip(obs, pred))
    ax.text(0.05, 0.90, f"最大偏差 = {dev:.1e}", transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", lw=0.6))
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("解析值 Ξᴮᴺ")
    ax.set_ylabel("合成实测 Ξᴮᴺ")
    ax.set_title("(a) 受控合成验证（11 组配置）", fontsize=9)
    # (b) 真实数据
    ax = axes[1]
    def rho_curve(Delta, xi):
        gamma = np.sqrt(1 + Delta**2 / 4) / np.sqrt(1 + (Delta - xi)**2 / 4)
        return (Delta - (Delta - xi) * gamma) / xi
    real = [
        (0.9227, 0.4322, 2.3273, "#4C72B0", "deepseek"),
        (1.0828, 0.4469, 2.3273, "#4C72B0", "glm"),
        (2.8778, 0.3674, 5.0437, "#8172B3", "deepseek"),
        (2.8411, 0.3663, 5.0437, "#8172B3", "glm"),
        (2.9026, 0.5058, 4.4453, "#DD8452", "deepseek"),
        (2.7811, 0.5165, 4.2940, "#DD8452", "glm")]
    for Delta, color, lab in [(2.3273, "#4C72B0", "15 维（Δ = 2.33）"),
                              (5.0437, "#8172B3", "字符 2-4gram（Δ = 5.04）"),
                              (4.4453, "#DD8452",
                               "RoBERTa（Δ = 4.29~4.45，反解拟合值）")]:
        xs = np.linspace(0.05, Delta * 0.99, 200)
        ax.plot(xs, rho_curve(Delta, xs), color=color, lw=1.2, label=lab, zorder=2)
    for xi, ro, Delta, color, tgt in real:
        ax.scatter(xi, ro, s=34, marker=TGT_MARKER[tgt], color=color, zorder=4,
                   linewidths=0.4, edgecolors="white")
    ax.scatter([], [], s=30, marker="o", color="#666666", label="deepseek")
    ax.scatter([], [], s=30, marker="^", color="#666666", label="glm")
    ax.set_xlabel("初始判别漂移 Ξ_eff")
    ax.set_ylabel("BN 残留比 ρ = Ξᴮᴺ/Ξ_eff")
    ax.set_title("(b) 真实数据：解析曲线与实测", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left", handlelength=1.4)
    ax.set_xlim(0, 3.3); ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=8.5)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig3_bn_bound")

# ============================================================
# 图 4  主结果：零异源数据下的跨 LLM 校正（E26 / E29-roberta）
# ============================================================
def fig4():
    e26 = J("E26_tta.json"); e29r = J("_E29_roberta.json")
    e30 = J("E30_tta_select.json")
    panels = [
        ("15D", "15 维", e26["feature_sets"]["15D"]["alpha_0"],
         ["deepseek", "qwen", "glm", "gpt-4o", "llama3"]),
        ("char_2_4", "字符 2-4gram", e26["feature_sets"]["char_2_4"]["alpha_0"],
         ["deepseek", "qwen", "glm", "gpt-4o", "llama3"]),
        ("roberta", "RoBERTa（768 维冻结）", e29r["feature_sets"]["roberta"]["alpha_0"],
         ["deepseek", "glm"])]
    methods = [("none", "无校正", "#888888"), ("bn", "+BN", "#4C72B0"),
               ("ubc", "+UBC", "#C44E52")]

    def val(fs, blk, t, m):
        if m == "ubc":
            if fs == "roberta":
                return e29r["feature_sets"]["roberta"]["alpha_0"][t]["conf_0.30"]["F1_macro"]
            return e30["feature_sets"][fs]["alpha_0"][t + "_choice"]["oracle_F1"]
        return blk[t][m]["F1_macro"]

    tlab = {"deepseek": "Deep\nSeek", "qwen": "Qwen", "glm": "GLM",
            "gpt-4o": "GPT-\n4o", "llama3": "Lla\nma3"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, (fs, title, blk, tgts) in zip(axes, panels):
        x = np.arange(len(tgts)); w = 0.26
        for i, (m, lab, c) in enumerate(methods):
            vals = [val(fs, blk, t, m) for t in tgts]
            ax.bar(x + (i - 1) * w, vals, width=w * 0.92, color=c,
                   label=lab, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([tlab[t] for t in tgts], fontsize=7.5)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0.70, 1.02)
        ax.grid(axis="y", lw=0.4, color="#dddddd", zorder=0)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("macro-F1（α = 0，零异源数据）")
    axes[-1].legend(fontsize=8, loc="lower right", ncol=1)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig5_tta_main")

# ============================================================
# 图 5  TTA 增益 vs 初始 Ξ_eff（E26，30 组）
# ============================================================
def fig5():
    e26 = J("E26_tta.json")
    pts = []
    for fs in e26["feature_sets"]:
        for al in e26["feature_sets"][fs]:
            blk = e26["feature_sets"][fs][al]
            for t in blk:
                if not isinstance(blk[t], dict) or "none" not in blk[t]:
                    continue
                xi = blk[t]["none"]["Xi_eff"]
                gain = blk[t]["em"]["F1_macro"] - blk[t]["none"]["F1_macro"]
                pts.append((xi, gain, al))
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    k, b = np.polyfit(xs, ys, 1)
    r = np.corrcoef(xs, ys)[0, 1]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    amap = {"alpha_0": ("o", "#4C72B0", "α = 0（仅 ChatGPT）"),
            "alpha_0.05": ("s", "#DD8452", "α = 0.05"),
            "alpha_0.2": ("^", "#55A868", "α = 0.2")}
    for al, (mk, c, lab) in amap.items():
        sel = [p for p in pts if p[2] == al]
        ax.scatter([p[0] for p in sel], [p[1] for p in sel], s=30, marker=mk,
                   color=c, linewidths=0.4, edgecolors="white", zorder=3, label=lab)
    xline = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(xline, k * xline + b, color="#333333", lw=1.0, ls="--", zorder=2)
    ax.axhline(0, color="#999999", lw=0.6)
    ax.text(0.60, 0.06, f"r = {r:.3f},  R² = {r**2:.3f},  n = {len(pts)}",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", lw=0.6))
    ax.set_xlabel("初始判别漂移 Ξ_eff（校正前）")
    ax.set_ylabel("EM 校正增益（macro-F1）")
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(labelsize=8.5)
    fig.tight_layout()
    save(fig, "fig6_gain_vs_xi")

# ============================================================
# 图 6  λ 敏感性 + 批次规模（E28 / E31）
# ============================================================
def fig6():
    e28 = J("_E28_i1.json"); e31 = J("E31_batch_size.json")
    e32 = J("E32_roberta_batch.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    style = {("15D", "deepseek"): ("#4C72B0", "o", "15 维 / DeepSeek"),
             ("15D", "glm"): ("#4C72B0", "^", "15 维 / GLM"),
             ("char_2_4", "deepseek"): ("#C44E52", "o", "字符 2-4gram / DeepSeek"),
             ("char_2_4", "glm"): ("#C44E52", "^", "字符 2-4gram / GLM"),
             ("roberta", "deepseek"): ("#55A868", "o", "RoBERTa / DeepSeek"),
             ("roberta", "glm"): ("#55A868", "^", "RoBERTa / GLM")}

    def gain_of(fs, t, n):
        if fs == "roberta":
            return e32["feature_sets"]["roberta_frozen"][t][f"n_{n}"]["gain"]
        return e31["feature_sets"][fs][t][f"n_{n}"]["gain"]

    # (a) lambda
    ax = axes[0]
    for (fs, t), (c, mk, lab) in style.items():
        if fs == "roberta":
            continue          # λ 消融未覆盖 RoBERTa（E28 仅 15D / char）
        blk = e28["feature_sets"][fs]["alpha_0"][t]
        ys = [blk[f"lam_{l:g}"]["F1_macro"] for l in lams]
        ax.plot(lams, ys, color=c, marker=mk, ms=4, lw=1.1,
                ls="-" if t == "deepseek" else "--", label=lab, zorder=3)
    ax.set_xlabel("位移系数 λ（单步实现）")
    ax.set_ylabel("macro-F1")
    ax.set_title("(a) λ 敏感性", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(labelsize=8.5)
    # (b) batch size
    ax = axes[1]
    ns = e31["sizes"]
    for (fs, t), (c, mk, lab) in style.items():
        ys = [gain_of(fs, t, n) for n in ns]
        ax.plot(ns, ys, color=c, marker=mk, ms=4, lw=1.1,
                ls="-" if t == "deepseek" else "--", label=lab, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns], fontsize=8)
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_xlabel("批次规模 n（对数轴）")
    ax.set_ylabel("UBC 增益（macro-F1）")
    ax.set_title("(b) 批次规模敏感性", fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.set_ylim(-0.025, 0.235)
    ax.legend(fontsize=6.5, loc="upper right", ncol=2, framealpha=0.95,
              handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig7_lambda_batch")

# ============================================================
# 图 7  CLIDA γ 扫描（E20）
# ============================================================
def fig7():
    e20 = J("_E20_v2.json")
    gammas = [0, 1, 3]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    style = {("15D", "deepseek"): ("#4C72B0", "o", "15 维 / DeepSeek"),
             ("15D", "glm"): ("#4C72B0", "^", "15 维 / GLM"),
             ("char_2_4", "deepseek"): ("#C44E52", "o", "字符 2-4gram / DeepSeek"),
             ("char_2_4", "glm"): ("#C44E52", "^", "字符 2-4gram / GLM")}
    panels = [(axes[0], "clida", "(a) CLIDA 投影"),
              (axes[1], "diag_fisher", "(b) 对角 Fisher 白化（对照）")]
    for ax, method, title in panels:
        for (fs, t), (c, mk, lab) in style.items():
            blk = e20["feature_sets"][fs]["alpha_0"]
            ys = [blk[f"gamma_{g}"][t][method]["F1_macro"] for g in gammas]
            ax.plot(gammas, ys, color=c, marker=mk, ms=4.5, lw=1.1,
                    ls="-" if t == "deepseek" else "--", label=lab, zorder=3)
            base = blk["gamma_0"][t]["raw_lr"]["F1_macro"]
            ax.axhline(base, color=c, lw=0.7, ls=":", alpha=0.55, zorder=1)
        ax.set_xticks(gammas)
        ax.set_xlabel("不变惩罚系数 γ")
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8.5)
    axes[0].set_ylabel("macro-F1（α = 0）")
    axes[0].set_ylim(0.755, 0.870)
    axes[1].legend(fontsize=7.5, loc="lower center", ncol=2,
                   handlelength=1.2, columnspacing=0.9)
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig9_clida_gamma")

# ============================================================
# 图 8  S_E 与 S_B 的重合（E20 scatter + 表 27）
# ============================================================
def fig8():
    e20 = J("_E20_v2.json")
    s15 = e20["feature_sets"]["15D"]["alpha_0.2"]["scatter"]
    sch = e20["feature_sets"]["char_2_4"]["alpha_0.2"]["scatter"]
    maincos = {"15D": 0.841, "char_2_4": 0.773}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    x = np.arange(2); w = 0.35
    # (a) 迹比
    ax = axes[0]
    r1 = [s15["tr_S_E"] / s15["tr_S_B"], sch["tr_S_E"] / sch["tr_S_B"]]
    r2 = [s15["tr_S_E"] / s15["tr_S_W"], sch["tr_S_E"] / sch["tr_S_W"]]
    ax.bar(x - w / 2, r1, width=w, color="#C44E52",
           label="tr(SE) / tr(SB)", zorder=3)
    ax.bar(x + w / 2, r2, width=w, color="#4C72B0",
           label="tr(SE) / tr(SW)", zorder=3)
    for xi, v in zip(x - w / 2, r1):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, r2):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["15 维", "字符 2-4gram"], fontsize=8.5)
    ax.set_ylabel("散度迹之比（α = 0.2）")
    ax.set_title("(a) 总量：环境散度与类间散度同量级", fontsize=8.5)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_ylim(0, 1.95)
    ax.tick_params(labelsize=8.5)
    # (b) 主余弦
    ax = axes[1]
    vals = [maincos["15D"], maincos["char_2_4"]]
    ax.bar(x, vals, width=0.45, color=["#4C72B0", "#8172B3"], zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.axhline(1.0, color="#999999", lw=0.7, ls="--")
    ax.text(1.42, 1.008, "完全重合", fontsize=7.5, color="#777777", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(["15 维", "字符 2-4gram"], fontsize=8.5)
    ax.set_ylabel("SE 与 SB 前 3 主方向最大主余弦")
    ax.set_title("(b) 方向：主子空间高度重合", fontsize=8.5)
    ax.set_ylim(0, 1.1)
    ax.tick_params(labelsize=8.5)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig10_se_sb")

# ============================================================
# 图 9  校正族：精度 vs 残留漂移（E29）
# ============================================================
def fig9():
    e29 = J("_E29_fast.json")
    order = ["bn", "bnc", "em", "trim_0.30", "conf_0.30", "trim_0.50",
             "conf_0.50", "trim_0.70", "conf_0.70", "oracle"]
    labels = ["BN", "BNC", "EM", "trim\n.3", "conf\n.3", "trim\n.5",
              "conf\n.5", "trim\n.7", "conf\n.7", "oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, fs, title in [(axes[0], "15D", "(a) 15 维 / DeepSeek"),
                          (axes[1], "char_2_4", "(b) 字符 2-4gram / DeepSeek")]:
        blk = e29["feature_sets"][fs]["alpha_0"]["deepseek"]
        f1 = [blk[m]["F1_macro"] for m in order]
        rho = [blk[m]["resid"] for m in order]
        colors = ["#4C72B0", "#4C72B0", "#C44E52", "#DD8452", "#DD8452",
                  "#DD8452", "#DD8452", "#DD8452", "#DD8452", "#55A868"]
        x = np.arange(len(order))
        ax.bar(x, f1, width=0.62, color=colors, zorder=3)
        for xi, v in zip(x, f1):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.2)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.8)
        ax.set_ylim(0.5, 1.06)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
        ax2 = ax.twinx()
        ax2.plot(x, rho, color="#333333", marker="D", ms=3.5, lw=1.0, zorder=4)
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(labelsize=8)
        if fs == "15D":
            ax.set_ylabel("macro-F1（柱，左轴）")
            ax2.set_ylabel("残留漂移比 ρ（线，右轴）")
        else:
            ax2.set_ylabel("残留漂移比 ρ（线，右轴）", fontsize=8)
    fig.text(0.5, -0.03, "蓝 = 批对齐族，红 / 橙 = 类条件对齐与截断族，绿 = 真值上限",
             ha="center", fontsize=8)
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig4_correction_family")

# ============================================================
# 图 8（正文）  真实改写攻击下的 TTA-EM 鲁棒性（E33）
# ============================================================
def fig_para(json_name="E33b_paraphrase_diag.json"):
    e = J(json_name)
    conds = e["conds"]
    targets = ["deepseek", "glm"]
    c_lab = {"orig": "无攻击", "para_ai": "仅 AI 侧被改写", "para_all": "双侧均被改写"}
    t_col = {"deepseek": "#4C72B0", "glm": "#DD8452", "qwen": "#55A868"}
    t_lab = {"deepseek": "DS", "glm": "GLM", "qwen": "QW"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, fs in [(axes[0], "15D"), (axes[1], "char_2_4")]:
        if fs not in e["feature_sets"]:
            continue
        groups = [(c, t) for c in conds for t in targets]
        x = np.arange(len(groups))
        w = 0.36
        f1n = [e["feature_sets"][fs]["targets"][t][c]["F1_none"] for c, t in groups]
        f1e = [e["feature_sets"][fs]["targets"][t][c]["F1_em"] for c, t in groups]
        cols = [t_col.get(t, "#4C72B0") for _, t in groups]
        ax.bar(x - w / 2, f1n, width=w, color="#C9CBCF", zorder=3,
               label="无校正" if fs == "15D" else None)
        ax.bar(x + w / 2, f1e, width=w, color=cols, zorder=3,
               label="TTA-EM" if fs == "15D" else None)
        for xi, gn, ge in zip(x, f1n, f1e):
            ax.text(xi + w / 2, ge + 0.012, "%+.3f" % (ge - gn),
                    ha="center", fontsize=5.8, color="#333333")
        lo = max(0.0, min(f1n + f1e) - 0.08)
        ax.set_ylim(lo, 1.06)
        ax.set_xticks(x)
        ax.set_xticklabels([t_lab.get(t, t) for _, t in groups], fontsize=7.5)
        # 条件标签画在坐标区下方（避免与柱体/刻度标签重叠）
        for ci, c in enumerate(conds):
            mid = ci * len(targets) + (len(targets) - 1) / 2.0
            ax.text(mid, -0.19, c_lab.get(c, c), ha="center", fontsize=7.5,
                    color="#333333", transform=ax.get_xaxis_transform(), clip_on=False)
        for b in [0.5 + len(targets) * i for i in range(len(conds) - 1)]:
            ax.axvline(b, color="#CCCCCC", lw=0.6, zorder=0)
        ax.grid(axis="y", lw=0.4, color="#DDDDDD", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8.5)
        ax.set_title(("(a) " if fs == "15D" else "(b) ") + FS_LABEL.get(fs, fs),
                     fontsize=9)
    axes[0].set_ylabel("macro-F1")
    if "15D" in e["feature_sets"]:
        from matplotlib.patches import Patch
        h = [Patch(fc="#C9CBCF", label="无校正"),
             Patch(fc=t_col.get(targets[0], "#4C72B0"), label="TTA-EM（%s）" % t_lab.get(targets[0], targets[0])),
             Patch(fc=t_col.get(targets[1], "#DD8452"), label="TTA-EM（%s）" % t_lab.get(targets[1], targets[1]))]
        fig.legend(handles=h, fontsize=7.2, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 1.0), handlelength=1.1,
                   handleheight=0.9, columnspacing=1.4, frameon=False)
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.92))
    save(fig, "fig8_paraphrase_attack")


if __name__ == "__main__":
    for fn in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig_para]:
        fn()
    print("ALL DONE")
