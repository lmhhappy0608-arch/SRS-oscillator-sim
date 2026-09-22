# Classification experiments

这一层是五态分类的观察与试验，**还没有替换** `code/plot_rotation_vs_frequency.py` 里的旧 `classify_state()`。

## 脚本

| 文件 | 作用 |
|---|---|
| `code/observe_plateaus.py` | 当前：从原始 \(\rho\) 估 \(c,\sigma,f\) |
| `code/fit_rotation_mixture.py` | 三线混合（停 / 锁 / 漂） |
| `code/analyze_mixture_residuals.py` | 残差诊断 |
| `code/plot_weight_simplex.py` | 混合权重单纯形 |

在 `classification/` 下运行：

```bash
python3 code/observe_plateaus.py
```

原始数值解仍读仓库根目录的 `raw_trials/`。
