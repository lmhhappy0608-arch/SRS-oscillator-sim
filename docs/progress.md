# 研究进度

按时间记「做过什么、得到什么、下一刀是什么」。后面继续往末尾加节，不要另开仓库。

## 1. 复现模型

Winfree 模型放在 ER 网上，扫 \(K\)、\(\delta\)、连边概率 \(p\)。

- 代码：`code/Ariaratnam2001.py`，网格生成 `code/generate_raw_grid.py`
- 参数：\(N=1000\)，典型 \(T=10000\)，\(T_0=8000\)
- 产出：各 \(p\) 层相图 `winfree_phase_diagram_p_*.png`

早期用宏观序参量 \(R\) 看同步强弱：密网、窄频更容易同步。平均 \(R\) 会把不同质的结局混在一起。

## 2. 从 \(R\) 转到 rotation number

每个振子的转速

\[
\rho_i=\frac{\theta_i(T)-\theta_i(T_0)}{T-T_0}.
\]

看 \(\rho(\omega)\) 才能区分停、锁、漂。作图：`code/plot_rotation_vs_frequency.py`。

## 3. 旧的五种状态分类

规则写在 `classify_state()`，**现在还在用，没有替换**：

| 状态 | 旧规则（摘要） |
|---|---|
| Death | \(\|\rho\|<0.01\) 的比例 \(\ge 0.95\) |
| Partial Death | 按 \(\omega\) 排序后连续至少 4 个靠近 0 |
| Locking | 全体 \(\rho\) 几乎一样，或非零平台覆盖 \(\ge 0.95\) |
| Partial Locking | 找到非零平台（连续 \(\ge 15\) 个、\(\rho\) 跨度 \(\le 0.02\)） |
| Incoherence | 没有这样的平台 |

问题：连续个数、0.02、0.95 都是预设。Lock 的中心 \(c\) 事先不知道，Death 才知道要围着 0。

## 4. 数值交叉验证

C++ RK4 / RK45 与 Python 对照，见 `cpp/`。确认 rotation number 不是求解器假象。

## 5. 三线混合（还没有成为新分类器）

每个振子比离三条几何线谁近：\(\rho=0\)（停）、\(\rho=c\)（水平锁）、\(\rho\approx\omega\)（漂）。

- 代码：`classification/code/fit_rotation_mixture.py`
- 残差：`classification/code/analyze_mixture_residuals.py`
- 计划备忘：`classification/MIXTURE_STATS_PLAN.md`

格子级得到权重 \((w_{\mathrm{death}},w_{\mathrm{lock}},w_{\mathrm{drift}})\)。这是观察工具，**没有改 `classify_state()`**。

## 6. 当前：观察 plateau，估 \(c\) 和 noise

教授的要求：先看振子行为，不要预设「连续几个在某个范围」；Death 围着 0，Lock 的 \(c\) 未知；noise 从结果里读。

- 代码：`classification/code/observe_plateaus.py`
- 图：`classification/figures/plateau_observe/`
- 表：`classification/tables/plateau_observe_gallery.csv`

做法：在 \(\rho\) 上找紧密簇（不要求 \(\omega\) 连续），

- \(c=\) 簇的中位数
- \(\sigma=\) 簇的标准差（当前 \(\Delta T=2000\) 下的 noise）
- \(f=\) 进簇比例

\(p=0.8\)、trial 0 的观察量级：

- 纯 Death：\(c=0\)，\(\sigma\sim 10^{-10}\)
- 纯 Lock：\(c\) 随 \(K\) 变（例如 0.93 与 0.76），\(\sigma\sim 10^{-5}\)
- Partial：同一条带，只是 \(f<1\)
- Incoherence：没有紧密水平簇

图上的 `death` / `lock` 字样只是选图菜单，不是新分类标签。

## 7. 下一步（还没做）

- 固定几个已看见平台的 trial，加长 \(T\)，看 \(\sigma\) 是否变小
- 用观察结果再写 plateau 定义，再考虑要不要动五态分类器
