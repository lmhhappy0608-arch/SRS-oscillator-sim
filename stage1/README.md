# Stage 1 exploratory parameter scan

This package generates rotation-number profiles for the Winfree model on
undirected ER networks. It does not classify Death, Partial Death, Locking,
Partial Locking, or Incoherence.

The default solver is fixed-step RK4 from `code/generate_raw_grid.py`
(`winfree_rhs` and `rk4_step`). The mathematical definition is the production
one shared with `code/Ariaratnam2001.py` and `cpp/main.cpp`:

```text
dθ/dt = ω - (K/N) sin(θ) ⊙ (A @ (1 + cos(θ)))
ω_i = linspace(1 - Δ, 1 + Δ, N)
```

`A` has no self-loops. At `p = 1` the graph is complete and the factor is still
`K/N`; it is not rewritten as `K/(N-1)` and the diagonal stays zero.

`code/Winfree_phase_3d.py` draws an asymmetric adjacency for `p < 1`. The C++
driver consumes randomness as network-then-phases and uses a different seed
mixer. Stage 1 does not change those files. Its own seeds are separate SHA-256
streams, so paired `(K, p, Δ)` points can share inputs. Natural frequencies are
deterministic; `frequency_seed` is null. A random-frequency mode is refused.

RK45 is available only as a fresh integration through
`Ariaratnam2001.integrate_winfree_system`. Midpoint continuation of RK45 raises
an error because the adaptive controller is not stored. RK4 continuation
restarts at the last completed full step (`theta_grid`, `step_index`), never
at a newly drawn initial condition and never at the shortened final step.

## Layout

```text
stage1_runs/                  # configured output directory
  index.sqlite                 # query this, not the NPZ tree
  runs/ab/<run_id>.npz
  runs/ab/<run_id>.json
  runs/ab/<run_id>.checkpoint.npz
```

`omega[i]` and every rotation profile at column `i` are the same oscillator.
Saved arrays stay in that order. `oscillator_index` is `0 .. N-1`.

## Commands

Subgrid first. The default config is the full candidate grid
(K step 0.01, 11 values of p, Delta step 0.05, 3 trials, T = 14000) and
`scan` will start it.

```bash
python3 -m stage1 scan --config stage1/configs/stage1_default.json \
  --k-values 0.70 --p-values 0.80 --delta-values 0.20 --trials 1 --workers 1
```

Stop with Ctrl-C. The current run is left as `checkpoint` when a checkpoint
file exists. Start the same command again to resume. Completed runs are skipped
and their files are not rewritten.

A run that was marked running but has no checkpoint is not recomputed unless
you pass `--allow-incomplete-restart`.

Extend a finished RK4 trajectory from its last full step:

```bash
python3 -m stage1 continue --output-dir stage1_runs --run-id RUN_ID \
  --t-end 16000 --window 14000 16000
```

Times that were never snapshotted and that lie before the checkpoint are
refused. The parent NPZ is left unchanged. The extended run gets a new
`run_id` and records `parent_run_id`.

## Pilot grid

`stage1/configs/stage1_pilot.json` now plans only the new p=0.2 cells:
K from 3.05 to 8 with step 0.05 (100 values), and Delta from 0 to 1 with
step 0.025 (41 values), one paired trial, 4100 runs. K=3 is already stored,
so it is not in this grid. Results still go to `stage1_runs`. A completed
`run_id` is skipped and its files are not rewritten.

```bash
python3 -m stage1 scan --config stage1/configs/stage1_pilot.json --workers 1
```

Progress reads the index and does not integrate:

```bash
python3 -m stage1 inspect --config stage1/configs/stage1_pilot.json
python3 -m stage1 inspect --config stage1/configs/stage1_pilot.json --check-files
```

Ctrl-C leaves a checkpoint. The same `scan` command resumes and skips
completed runs.

One saved profile, with the pairwise window differences:

```bash
python3 -m stage1 inspect --config stage1/configs/stage1_pilot.json \
  --K 0.70 --p 0.80 --Delta 0.20 --trial 0 \
  --plot stage1_pilot_plots/rho_K0.70_p0.80_Delta0.20.png
```

The plot sorts a copy by frequency. The NPZ arrays stay in the original
oscillator order. The command does not assign five-state labels.

Load the same arrays for an external classifier:

```python
from stage1 import query_runs, load_profiles

rows = query_runs(K=0.70, p=0.80, Delta=0.20, N=1000, trial_id=0)
records = load_profiles([row["run_id"] for row in rows if row["status"] == "completed"])
record = records[0]
record["omega"]
record["rotation_numbers"]
record["parameters"]
record["realization"]
```

## Pilot exploration

Descriptive features, parameter-space maps, and a review sample of saved
pilot profiles. This does not integrate or label states.

```bash
python3 -m stage1.explore
```

Output goes to `stage1_analysis/pilot_exploration/`. See that directory's
README after the script has been run.

## Targeted refinement

Copy `stage1/configs/stage1_refine.example.json`, edit the K interval, step,
p, Delta, and `n_trials`, and keep `output_dir` as `stage1_runs`. Points
whose parameters, seeds, solver, step size, horizon, and windows already
match a completed run are skipped. `n_trials` starts at `trial_id` 0, so
raising it adds later trials and keeps the original trial.

```bash
python3 -m stage1 inspect --config stage1/configs/stage1_refine.json
python3 -m stage1 scan --config stage1/configs/stage1_refine.json --workers 1
```

Changing `master_seed`, `N`, `dt`, `t_end`, the observation windows, or the
solver creates new `run_id`s.

## Query

```python
from stage1 import query_runs, load_run

runs = query_runs(K=0.70, p=0.80, Delta=0.20, N=1000)
profile = load_run(runs[0]["run_id"])
```

`profile["omega"]`, `profile["rotation_numbers"]` with shape `(windows, N)`,
`profile["order_parameter"]`, and `profile["metadata"]` are the stored record.

Randomness modes, set by `randomness_mode` or `--mode`:

| Mode | What changes across trials | What stays shared across K, p, Delta |
|---|---|---|
| `paired` | network id and initial-condition id together | the same ids; `p` thresholds one uniform field; `Δ` rescales one quantile index |
| `network` | network id only | initial phases and the frequency rule |
| `initial_condition` | initial phases only | the network id and the frequency rule |
| `independent` | both seeds, and the seeds also depend on K, p, Delta | nothing across parameter cells |

## Benchmark

Counting and extrapolation do not integrate:

```bash
python3 -m stage1 benchmark --preset all
```

Representative timings are a separate command. They are not the production scan:

```bash
python3 -m stage1 benchmark --measure --N 1000 --horizons 20 40 \
  --measure-output stage1_benchmark/measurements.json
python3 -m stage1 benchmark --preset dense \
  --measurements stage1_benchmark/measurements.json --workers 4
```

Wall-clock scaling stays empty until you pass a measured `--efficiency`.
The reused right-hand side is a dense matrix-vector product, so `p` may not
change the runtime much.

## Tests

```bash
python3 -m unittest stage1.tests.test_stage1 stage1.tests.test_scan stage1.tests.test_pilot
```

