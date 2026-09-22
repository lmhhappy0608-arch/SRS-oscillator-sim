# 三成分混合的统计细化计划

振子级仍然只用三条几何线：停 \(\rho=0\)、一条水平锁 \(\rho=c\)、漂 \(\rho\approx\omega\)。统计要回答两件事：这三条线在哪些 \((K,\delta)\) 格子不够；系统级如何用权重把 Death / Partial / Locking 划开，而不回到 \(0.95\) 这类阈值。

混合模型在某个权重为 0 时参数不可识别，\(2(\ell_1-\ell_0)\) 不再是标准 \(\chi^2\)。比较 2 / 3 / 4 条线时用 BIC 或 bootstrap，不要写普通似然比 \(p\) 值了事。

## 建议顺序

| 步 | 做什么 | 看什么 | 用来细化什么 |
| --- | --- | --- | --- |
| 1 | 每个格子算 RMSE、最大残差、残差对 \(\omega\) 的形状 | 高残差是否连成一带 | 哪些 \((K,\delta)\) 三线不够 |
| 2 | 权重 \((w_d,w_l,w_r)\) 画单纯形 / 熵 \(H(w)\) | 点是聚成 3 堆还是 5 堆 | Partial 是不是单独一团，还是连续带 |
| 3 | GMM / k-means 只作用在权重上，silhouette 选簇数 | 簇中心是否落在三个顶点和两条棱上 | 宏观五态的数据驱动划分 |
| 4 | 只对高残差格子加模型：第二条水平线，或斜率自由的斜枝 | \(\Delta\mathrm{BIC}<0\) 才保留 | slipping、双平台、弯枝 |
| 5 | 跨 \(p=0.2,\ldots,1.0\) 比较锁定区面积、混合带宽 | 随 \(p\) 是否单调变 | 网络密度怎么改相图 |

## 宏观态怎么从权重读出来

粗相：一个权重接近 1。Death / Locking / Incoherence 对应单纯形的三个顶点。不必再设 0.95；用权重空间里的聚类，或用「最大权重明显大于另外两个」这种由数据决定的分离。

Partial：落在一条棱上。Partial Death \(\approx\) death–drift 棱（\(w_{\mathrm{lock}}\approx 0\)，另外两个都明显）。Partial Locking \(\approx\) lock–drift 棱。混合程度就是较小那个权重，不再是「至少 15 个点」。

## 残差在说什么

按 \(\omega\) 排序后看残差形态：

| 残差形态 | 更像哪种漏网情况 | 下一步模型 |
| --- | --- | --- |
| 一条水平残差平台 | 第二条锁定转速 | 再加一条 \(\rho=c_2\) |
| 中间一段残差系统偏离 0 和 \(\omega\) | slipping / 频率被拉弯 | 斜段斜率自由 |
| 两端大、中间小 | 平台漏了，或 \(c\) 估偏 | 检查 `lock_center` |
| 处处小、无结构 | 三线已够 | 不再加成分 |

对照旧五色图时，不要把 \(\mathrm{argmax}(w)\) 当成新的五分类再做混淆矩阵。先对每个旧标签画 \((w_d,w_l,w_r)\) 的箱线图：Death 应贴 \((1,0,0)\)，Partial Death 应在 death–drift 棱上散开。若旧的 Partial Locking 在权重空间里不聚成一团，那是阈值造的色块，不是要保留的第五相。

## 目录与第一步怎么跑

分类相关代码、图和表都放在 `classification/`：

- 原始数值解：仓库根目录 `raw_trials/`（不进 git）
- 混合拟合：`classification/code/fit_rotation_mixture.py`
- 第一步残差：`classification/code/analyze_mixture_residuals.py`
- 混合权重图：`classification/figures/rotation_mixture/`
- 残差图：`classification/figures/mixture_residuals/`
- 表：`classification/tables/`

在 `classification/` 下：

```bash
python3 code/analyze_mixture_residuals.py
python3 code/analyze_mixture_residuals.py --p 0.2 0.4 0.6 0.8 1.0
```

每个格子会写出 RMSE、最大 \(|\rho-\hat\rho|\)、残差对 \(\omega\) 的 lag-1 自相关、低/中/高 \(\omega\) 三段 RMSE、残差的倾斜和弯曲。相图看高残差是否成带；例子图看残差对 \(\omega\) 的形状。
