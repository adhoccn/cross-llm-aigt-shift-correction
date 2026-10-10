[TABLE 27]
| 特征集 | 目标 | 原始 LR | `diag_heur`（本文前期调制） | `diag_fisher` | CLIDA $\gamma{=}0$ |
|---|---|---|---|---|---|
| char 2-4 | deepseek | 0.801 | 0.697 | 0.833 | 0.835 |
| char 2-4 | glm | 0.793 | 0.726 | 0.808 | 0.819 |
| 15D | deepseek | 0.852 | 0.854 | 0.845 | 0.854 |
| 15D | glm | 0.795 | 0.790 | 0.783 | 0.796 |

*注：$\alpha=0$，5-fold×5 seeds，macro-F1。*
