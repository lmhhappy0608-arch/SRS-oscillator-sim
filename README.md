# Winfree oscillators on ER networks (SRS)

ER 随机图上的 Winfree 模型：从复现 Ariaratnam & Strogatz (2001) 到用 rotation number 做五态分类，再到观察 plateau / noise。

当前进度：**在优化五种状态的分类，重点是重新观察 plateau 的定义。旧的 `classify_state()` 还没有被替换。**

更完整的时间线见 [docs/progress.md](docs/progress.md)。

## 现在做到哪

每个 trial 有 \(N=1000\) 个振子的 \((\omega,\rho)\)。观察脚本在 \(\rho\) 上找紧密水平簇，读出

- \(c\)：簇的中位数（Lock 的共同转速；Death 应在 0）
- \(\sigma\)：簇的宽度，也就是当前 \(T\) 下的 noise
- \(f\)：进这个簇的占比（用来区分 complete / partial）

分类器本身仍是 `code/plot_rotation_vs_frequency.py` 里的旧规则（`|\rho|<0.01`、连续 15 个点、覆盖 0.95 等）。

## 目录

| 路径 | 内容 |
|---|---|
| `code/` | Python 模拟、\(\rho(\omega)\) 作图、旧五态分类 |
| `cpp/` | C++ RK4 / RK45 交叉验证 |
| `classification/` | 三线混合、残差、plateau 观察（不改旧分类器） |
| `docs/progress.md` | 研究时间线，后面做了就往上加 |
| `tables/` | 清洗对照、混合权重等 CSV |
| `winfree_phase_diagram_p_*.png` | 旧规则下的 \(p\) 层相图 |

原始数值解 `raw_trials/` 体积大，不进 git。需要数据时在本地重跑网格。

## 怎么跑

**旧相图（复现 + 旧分类）**

```bash
python3 code/Ariaratnam2001.py --plot-stack --p-values 0.2 0.4 0.6 0.8
```

**单格子 \(\rho(\omega)\)**

```bash
python3 code/plot_rotation_vs_frequency.py --clean none --k 0.4 --delta 0.08 --p 0.8 --trial 0
```

**Plateau 观察（当前）**

```bash
cd classification
python3 code/observe_plateaus.py
python3 code/observe_plateaus.py --k 0.7 --delta 0.20 --p 0.8 --trial 0
```

图写到 `classification/figures/plateau_observe/`，表写到 `classification/tables/plateau_observe_gallery.csv`。

## 依赖

Python 3，NumPy，SciPy，Matplotlib。C++ 交叉验证见 `cpp/README.md`。
