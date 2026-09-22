# Winfree on ER networks (C++)

Ariaratnam & Strogatz (2001) Winfree model on ER random graphs. Integrates with **RK4** (default) or RK45 and writes `omega` and `rotation_number` for each trial (**no** phase-state vector).

## Layout

```
cpp/
├── main.cpp                    # parameter grid + ODE + npz writer
├── build.sh                    # recommended build (no CMake)
├── run_test.sh                 # 4-cell smoke test (includes --benchmark)
├── CMakeLists.txt              # optional CMake build
├── Makefile                    # optional make build
├── third_party/cnpy/           # NumPy .npz I/O
│   ├── cnpy.h
│   └── cnpy.cpp
└── build/
    └── generate_raw_grid       # compiled binary
```

## Build

**Recommended** (macOS / Linux):

```bash
cd cpp
./build.sh
```

Equivalent:

```bash
clang++ -std=c++17 -O3 -DNDEBUG \
  -I third_party/cnpy \
  main.cpp third_party/cnpy/cnpy.cpp -lz \
  -o build/generate_raw_grid
```

Needs Xcode Command Line Tools (`clang++`) and system `zlib` (`-lz`).

Other options:

```bash
make
mkdir build && cd build && cmake .. && cmake --build .
```

> Build with `./build.sh` or `make`. Do not compile `main.cpp` alone (it must link cnpy and zlib).

## Examples

### Smoke test (benchmark, seconds)

`--benchmark` sets `N=100, T=1000, T0=800`. Use it only to check the pipeline, **not** against production `T=10000` results.

```bash
./run_test.sh
```

Or four \((K,\delta)\) cells by hand:

```bash
./build/generate_raw_grid \
  --benchmark \
  --k-min 0 --k-max 0.4 --k-step 0.4 \
  --delta-min 0.10 --delta-max 0.15 --delta-step 0.05 \
  --p 0.8 --trials 1 \
  --output-dir raw_trials_cpp
```

Cells: \(K\in\{0.0,0.4\}\), \(\delta\in\{0.10,0.15\}\), **4** npz files.

### Full production grid (441 cells, same parameters as Python)

Drop `--benchmark` to use `N=1000, T=10000, T0=8000`.
\(K\in[0,2]\) step 0.1 \(\times\) \(\delta\in[0,1]\) step 0.05 = **441** cells. Serial (no OpenMP): about **2–3 minutes** per cell, **15–22 hours** for the grid.
Write to `raw_trials_c` so it stays separate from Python `raw_trials/`.

```bash
./build.sh
./build/generate_raw_grid \
  --k-min 0 --k-max 2 --k-step 0.1 \
  --delta-min 0 --delta-max 1 --delta-step 0.05 \
  --p 0.8 --trials 1 \
  --integrator rk4 \
  --output-dir raw_trials_c \
  --skip-existing
```

Startup log should show `integrator=rk4, dt=auto`. Each finished cell prints a timing such as `(168s)`.

### Small production subset

Same defaults `N=1000, T=10000, T0=8000`; about 2–3 minutes per cell.

```bash
./build/generate_raw_grid \
  --k-min 0 --k-max 0.4 --k-step 0.4 \
  --delta-min 0.10 --delta-max 0.15 --delta-step 0.05 \
  --p 0.8 --trials 1 \
  --output-dir raw_trials_cpp
```

### Custom size

```bash
./build/generate_raw_grid \
  --N 500 --T 5000 --T0 4000 \
  --k-min 0.5 --k-max 0.5 \
  --delta-min 0.2 --delta-max 0.2 \
  --p 0.8 --trials 3 \
  --output-dir raw_trials
```

## Performance

Default is **fixed-step RK4** (`--integrator rk4`), tens to hundreds of times faster than the original adaptive RK45.

| Integrator | Use | Speed |
|---|---|---|
| `rk4` (default) | Batch rotation numbers, enough accuracy | Fast |
| `rk45` | Strict match to SciPy `solve_ivp(RK45)` | Slow |

**Why the original RK45 was slow:**
- Error estimate used the wrong \(b_4\) (should be SciPy's E vector), so the step controller kept shrinking \(h\) toward ~1e-12
- 7 RHS evaluations per step (RK4 uses 4)
- Initial step 10 caused repeated rejections
- The controller kept shrinking even when the error was near the tolerance

**Suggested commands:**

```bash
# Fast benchmark (4 cells, about 1 s)
./build/generate_raw_grid --benchmark --k-min 0 --k-max 0.4 --k-step 0.4 \
  --delta-min 0.10 --delta-max 0.15 --delta-step 0.05 --p 0.8

# Production (N=1000, T=10000, about 1–3 min per cell)
./build/generate_raw_grid --integrator rk4 ...

# Align with Python (slow, verification only)
./build/generate_raw_grid --integrator rk45 ...
```

`--dt` applies to RK4 only. The default is chosen from the fastest natural frequency (about 0.1).

## OpenMP

Apple clang on macOS does not ship OpenMP. `build.sh` looks for Homebrew **libomp** (Apple Silicon: `/opt/homebrew/opt/libomp`; Intel: `/usr/local/opt/libomp`; also `brew --prefix libomp` if `brew` exists). Without libomp it still builds serially and prints a warning.

### Install and build

```bash
brew install libomp
cd cpp
./build.sh
```

A successful OpenMP build prints `OpenMP: enabled (libomp at ...)`.

### Thread count

- **`--workers N`**: OpenMP threads at run time (only if built with OpenMP)
- **`OMP_NUM_THREADS`**: used when `--workers` is omitted; `--workers` wins if both are set

```bash
# 2-thread benchmark
./build/generate_raw_grid --benchmark --workers 2 \
  --k-min 0 --k-max 0.4 --k-step 0.4 \
  --delta-min 0.10 --delta-max 0.15 --delta-step 0.05 \
  --p 0.8 --output-dir raw_trials_cpp

# Full grid: 8 threads + resume
OMP_NUM_THREADS=8 ./build/generate_raw_grid \
  --k-min 0 --k-max 2 --k-step 0.1 \
  --delta-min 0 --delta-max 1 --delta-step 0.05 \
  --p 0.8 --trials 1 \
  --integrator rk4 \
  --output-dir raw_trials_c \
  --skip-existing
```

The 441-cell grid is about **2–3 hours** on 8 cores, or 15–22 hours serial.

## CLI

| Flag | Default | Meaning |
|---|---|---|
| `--k-min/max/step` | 0 / 2 / 0.1 | Coupling \(K\) grid |
| `--delta-min/max/step` | 0 / 1 / 0.05 | Frequency width \(\delta\) grid |
| `--p` | 0.8 | ER edge probability |
| `--trials` | 1 | Repeats per cell |
| `--N` | 1000 | Number of oscillators |
| `--T` | 10000 | Integration end time |
| `--T0` | 8000 | Start time for the rotation number |
| `--benchmark` | off | Fast mode: N=100, T=1000, T0=800 |
| `--integrator` | rk4 | `rk4` or `rk45` |
| `--dt` | auto | Fixed RK4 step (from \(\omega\) by default) |
| `--output-dir` | raw_trials | Output directory |
| `--skip-existing` | off | Skip npz files that already exist |
| `--seed` | 0 | Global seed mixed into each trial seed |
| `--workers` | OpenMP default | OpenMP thread count |

## Output

```
{output_dir}/K_{K}_delta_{delta}_p_{p}_trial_{NN}.npz
```

Example: `raw_trials_cpp/K_0.40_delta_0.15_p_0.80_trial_00.npz`

| Field | Shape | Meaning |
|---|---|---|
| `omega` | (N,) float64 | Natural frequencies `linspace(1−δ, 1+δ, N)` |
| `rotation_number` | (N,) float64 | \((\theta(T)-\theta(T_0))/(T-T_0)\) |
| `K`, `delta`, `p` | scalar | Parameters |
| `trial`, `N`, `T`, `T0` | scalar | Metadata |

**No** phase-state label is stored. Classification is done later in Python `code/plot_rotation_vs_frequency.py`.

```python
import numpy as np
d = np.load("raw_trials_cpp/K_0.40_delta_0.15_p_0.80_trial_00.npz")
omega = d["omega"]
rotation = d["rotation_number"]
```

## Simulation flow (`main.cpp`)

For each \((K,\delta,\mathrm{trial})\) cell, in parallel:

1. **Seed** from `(K, δ, p, trial, global_seed)`
2. **Network**: undirected ER adjacency with probability \(p\)
3. **Frequencies**: \(\omega=\mathrm{linspace}(1-\delta,1+\delta,N)\)
4. **Initial data**: \(\theta_0\sim\mathrm{Uniform}(0,2\pi)\)
5. **Integrate** the Winfree ODE with RK4 (default, fixed step) or RK45 (SciPy `RK45`)
   ```
   dθ/dt = ω − (K/N) · sin(θ) ⊙ (A @ (1 + cos(θ)))
   ```
   Record \(\theta\) at \(T_0\) and \(T\)
6. **Rotation number**: `rotation_number[i] = (θᵢ(T) − θᵢ(T₀)) / (T − T₀)`
7. **Save** omega, rotation_number, and metadata (**no** \(\theta\) trajectory)

## Python post-processing

```bash
# Point RAW_DATA_DIR at the C++ output, or copy npz files into raw_trials/
python code/plot_rotation_vs_frequency.py --k 0.0 0.4 --delta 0.10 0.15 --p 0.8
```
