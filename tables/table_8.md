[TABLE 8]
| 模型 | 自变量 | $R^2$ | MAE |
|---|---|---|---|
| 单因子 | 仅残留漂移 $\Xi^{\mathrm{post}}_{\mathrm{eff}}$ | 0.2305 | 0.4991 |
| **双因子** | 残留漂移 + 注入方差 $\sigma^2_{\mathrm{inj}}$ | **0.6377** | **0.0295** |

*注：在 E30 的 66 个"变体×目标"样本点上回归 macro-F1 的实测值，覆盖 deepseek、qwen、glm 三个目标 LLM；数据源 results/_E30_pred3.json。*
