# Winfree ER 网络 C++ 仿真

Ariaratnam & Strogatz (2001) Winfree 模型，在 ER 随机图上用 **RK4**（默认）或 RK45 积分，输出每个 trial 的 `omega` 与 `rotation_number`（**不含** phase 状态向量）。

## 目录结构

```
cpp/
├── main.cpp                    # 主程序：参数网格 + ODE 积分 + 写 npz
├── build.sh                    # 推荐编译方式（无需 CMake）
├── run_test.sh                 # 4 格点小规模试跑（含 --benchmark）
├── CMakeLists.txt              # 可选：CMake 构建
├── Makefile                    # 可选：make 构建
├── third_party/cnpy/           # NumPy .npz 读写库
│   ├── cnpy.h
│   └── cnpy.cpp
└── build/
    └── generate_raw_grid       # 编译产物（可执行文件）
```

## 编译方法

**推荐**（macOS / Linux）：

```bash
cd /Users/mohan/Desktop/SRS/Cursor/cpp
./build.sh
```

等价命令：

```bash
clang++ -std=c++17 -O3 -DNDEBUG \
  -I third_party/cnpy \
  main.cpp third_party/cnpy/cnpy.cpp -lz \
  -o build/generate_raw_grid
```

依赖：Xcode Command Line Tools（`clang++`）、系统 `zlib`（`-lz`）。

其他方式：

```bash
make                    # 与 build.sh 类似
mkdir build && cd build && cmake .. && cmake --build .   # CMake
```

> **注意**：请用 `./build.sh` 或 `make` 编译，不要单独编译 `main.cpp`（需链接 cnpy 与 zlib）。

## 运行示例

### 快速验证（benchmark，秒级）

`--benchmark` 自动设置 `N=100, T=1000, T0=800`，仅用于验证流程，**不可**与正式 `T=10000` 结果对比。

```bash
./run_test.sh
```

或手动指定 4 个 (K, δ) 格点：

```bash
./build/generate_raw_grid \
  --benchmark \
  --k-min 0 --k-max 0.4 --k-step 0.4 \
  --delta-min 0.10 --delta-max 0.15 --delta-step 0.05 \
  --p 0.8 --trials 1 \
  --output-dir raw_trials_cpp
```

格点：K ∈ {0.0, 0.4}，δ ∈ {0.10, 0.15}，共 **4** 个 npz 文件。

### 正式全网格（441 格点，与 Python 参数一致）

去掉 `--benchmark` 即使用生产默认 `N=1000, T=10000, T0=8000`。  
K∈[0,2] step 0.1 × δ∈[0,1] step 0.05 = **441** 格点；串行（无 OpenMP）每格约 **2–3 分钟**，全网格约 **15–22 小时**。  
输出目录建议用 `raw_trials_c`，与 Python 的 `raw_trials/` 分开。

```bash
./build.sh   # 修改 main.cpp 后必须先编译
./build/generate_raw_grid \
  --k-min 0 --k-max 2 --k-step 0.1 \
  --delta-min 0 --delta-max 1 --delta-step 0.05 \
  --p 0.8 --trials 1 \
  --integrator rk4 \
  --output-dir raw_trials_c \
  --skip-existing
```

启动日志应显示 `integrator=rk4, dt=auto`；每格完成后会打印耗时，如 `(168s)`。

### 正式小规模参数组（few parameter groups）

去掉 `--benchmark` 即使用生产默认 `N=1000, T=10000, T0=8000`；每格约 2–3 分钟。

```bash
./build/generate_raw_grid \
  --k-min 0 --k-max 0.4 --k-step 0.4 \
  --delta-min 0.10 --delta-max 0.15 --delta-step 0.05 \
  --p 0.8 --trials 1 \
  --output-dir raw_trials_cpp
```

### 自定义规模

```bash
./build/generate_raw_grid \
  --N 500 --T 5000 --T0 4000 \
  --k-min 0.5 --k-max 0.5 \
  --delta-min 0.2 --delta-max 0.2 \
  --p 0.8 --trials 3 \
  --output-dir raw_trials
```

## 性能说明

默认使用 **固定步长 RK4**（`--integrator rk4`），比原 RK45 自适应积分快 **数十到数百倍**。

| 积分器 | 适用场景 | 相对速度 |
|--------|----------|----------|
| `rk4`（默认） | 批量生成 rotation number，精度足够 | 快 |
| `rk45` | 需与 SciPy `solve_ivp(RK45)` 严格对齐 | 慢 |

**原 RK45 慢的原因：**
- 误差估计用了错误的 b₄ 系数（应为 SciPy 的 E 向量），导致步长控制器误判，步长持续缩小至 ~1e-12
- 每步 7 次 RHS 求值（RK4 仅 4 次）
- 初始步长过大（10）导致反复拒绝
- 步长控制器在误差接近阈值时仍持续缩小步长

**推荐用法：**
```bash
# 快速 benchmark（4 格点，约 1 秒内）
./build/generate_raw_grid --benchmark --k-min 0 --k-max 0.4 --k-step 0.4 \
  --delta-min 0.10 --delta-max 0.15 --delta-step 0.05 --p 0.8

# 正式生产（N=1000, T=10000，每格约 1–3 分钟，视 N 与 dt 而定）
./build/generate_raw_grid --integrator rk4 ...

# 与 Python 对齐（慢，仅验证用）
./build/generate_raw_grid --integrator rk45 ...
```

`--dt` 仅对 RK4 有效；默认按最快自然频率自动选取（约 0.1）。

## 并行运行（OpenMP）

macOS 上 Apple clang 默认不带 OpenMP。`build.sh` 会自动检测 Homebrew 的 **libomp**（Apple Silicon：`/opt/homebrew/opt/libomp`；Intel Mac：`/usr/local/opt/libomp`；若已安装 `brew` 也会尝试 `brew --prefix libomp`）。未安装 libomp 时仍可串行编译，并打印警告。

### 安装与编译

```bash
brew install libomp
cd /Users/mohan/Desktop/SRS/Cursor/cpp
./build.sh
```

编译成功且启用 OpenMP 时，输出含 `OpenMP: enabled (libomp at ...)`。

### 指定线程数

- **`--workers N`**：运行时设置 OpenMP 线程数（仅编译时启用了 OpenMP 时有效）
- **`OMP_NUM_THREADS`**：环境变量，未指定 `--workers` 时使用；二者同时存在时 `--workers` 优先

```bash
# 2 线程 benchmark 验证并行
./build/generate_raw_grid --benchmark --workers 2 \
  --k-min 0 --k-max 0.4 --k-step 0.4 \
  --delta-min 0.10 --delta-max 0.15 --delta-step 0.05 \
  --p 0.8 --output-dir raw_trials_cpp

# 正式全网格：8 线程 + 断点续跑
OMP_NUM_THREADS=8 ./build/generate_raw_grid \
  --k-min 0 --k-max 2 --k-step 0.1 \
  --delta-min 0 --delta-max 1 --delta-step 0.05 \
  --p 0.8 --trials 1 \
  --integrator rk4 \
  --output-dir raw_trials_c \
  --skip-existing
```

441 格点全网格在 8 核并行下约 **2–3 小时**（视机器而定），串行约 15–22 小时。

## CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--k-min/max/step` | 0 / 2 / 0.1 | 耦合强度 K 网格 |
| `--delta-min/max/step` | 0 / 1 / 0.05 | 频率展宽 δ 网格 |
| `--p` | 0.8 | ER 连边概率 |
| `--trials` | 1 | 每格重复次数 |
| `--N` | 1000 | 振子数量 |
| `--T` | 10000 | 积分终止时间 |
| `--T0` | 8000 | 计算 rotation number 的起始时刻 |
| `--benchmark` | off | 快速模式：N=100, T=1000, T0=800 |
| `--integrator` | rk4 | ODE 积分器：`rk4` 或 `rk45` |
| `--dt` | auto | RK4 固定步长（默认按 ω 自动） |
| `--output-dir` | raw_trials | 输出目录 |
| `--skip-existing` | off | 跳过已存在的 npz（断点续跑） |
| `--seed` | 0 | 全局种子（混入每个 trial 的独立 seed） |
| `--workers` | OMP 默认 | OpenMP 线程数（需编译时启用 OpenMP） |

## 输出格式

文件名：

```
{output_dir}/K_{K}_delta_{delta}_p_{p}_trial_{NN}.npz
```

示例：`raw_trials_cpp/K_0.40_delta_0.15_p_0.80_trial_00.npz`

| 字段 | 形状 | 说明 |
|------|------|------|
| `omega` | (N,) float64 | 自然频率 linspace(1−δ, 1+δ, N) |
| `rotation_number` | (N,) float64 | (θ(T) − θ(T₀)) / (T − T₀) |
| `K`, `delta`, `p` | 标量 | 参数 |
| `trial`, `N`, `T`, `T0` | 标量 int | 元数据 |

**不含** phase 分类标签；分类在 Python `code/plot_rotation_vs_frequency.py` 中完成。

Python 读取示例：

```python
import numpy as np
d = np.load("raw_trials_cpp/K_0.40_delta_0.15_p_0.80_trial_00.npz")
omega = d["omega"]
rotation = d["rotation_number"]
```

## Simulation 流程（main.cpp）

对每个 (K, δ, trial) 格点，并行执行：

1. **种子**：由 `(K, δ, p, trial, global_seed)` 哈希得到 trial 级 seed  
2. **网络**：生成 ER 邻接表（概率 `p`）  
3. **频率**：`ω = linspace(1−δ, 1+δ, N)`  
4. **初值**：θ₀ ~ Uniform(0, 2π)  
5. **积分**：RK4（默认，固定步长）或 RK45（同 SciPy `RK45`）积分 Winfree 方程  
   ```
   dθ/dt = ω − (K/N) · sin(θ) ⊙ (A @ (1 + cos(θ)))
   ```
   固定/自适应步长，记录 T₀ 与 T 时刻的 θ  
6. **旋转数**：`rotation_number[i] = (θᵢ(T) − θᵢ(T₀)) / (T − T₀)`  
7. **保存**：`save_trial_npz` 写入 omega、rotation_number 及元数据（**不含** θ 轨迹）

## 与 Python 后处理对接

```bash
cd /Users/mohan/Desktop/SRS/Cursor
# 将 plot 脚本的 RAW_DATA_DIR 指向 cpp/raw_trials_cpp，或复制 npz 到 raw_trials/
python code/plot_rotation_vs_frequency.py --k 0.0 0.4 --delta 0.10 0.15 --p 0.8
```
