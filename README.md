# Winfree oscillators on ER networks (SRS)

Winfree model on ER random graphs: from reproducing Ariaratnam & Strogatz (2001), through five-state classification on rotation numbers, to observing plateaus and noise.

**Current status:** we are refining the five-state classification, with the plateau definition under review. The old `classify_state()` has not been replaced.

See [docs/progress.md](docs/progress.md) for the full timeline.

## Where things stand

Each trial has \(N=1000\) oscillators and a pair \((\omega,\rho)\). The observation script finds tight horizontal clusters in \(\rho\) and reports

- \(c\): median of the cluster (common locked rotation number; Death should sit at 0)
- \(\sigma\): width of the cluster, i.e. noise at the current \(T\)
- \(f\): fraction of oscillators in the cluster (complete vs partial)

The classifier itself is still the old rule set in `code/plot_rotation_vs_frequency.py` (\(|\rho|<0.01\), at least 15 consecutive oscillators, coverage 0.95, and so on).

## Layout

| Path | Contents |
|---|---|
| `code/` | Python simulation, \(\rho(\omega)\) plots, old five-state classifier |
| `cpp/` | C++ RK4 / RK45 cross-check |
| `classification/` | Three-line mixture, residuals, plateau observation (does not replace the old classifier) |
| `docs/progress.md` | Research timeline; append new sections here |
| `tables/` | Cleaning comparisons and mixture-weight CSVs |
| `winfree_phase_diagram_p_*.png` | Phase diagrams under the old rules, one slice per \(p\) |

Raw solutions in `raw_trials/` are large and not in git. Re-run the grid locally when you need them.

## How to run

**Old phase diagrams (reproduction + old labels)**

```bash
python3 code/Ariaratnam2001.py --plot-stack --p-values 0.2 0.4 0.6 0.8
```

**Single-cell \(\rho(\omega)\)**

```bash
python3 code/plot_rotation_vs_frequency.py --clean none --k 0.4 --delta 0.08 --p 0.8 --trial 0
```

**Plateau observation (current).** Full run guide: [classification/README.md](classification/README.md).

```bash
cd classification
python3 code/observe_plateaus.py
python3 code/observe_plateaus.py --k 0.7 --delta 0.20 --p 0.8 --trial 0
```

Figures go to `classification/figures/plateau_observe/`. The table is `classification/tables/plateau_observe_gallery.csv`.

## Requirements

Python 3 with NumPy, SciPy, and Matplotlib. See `cpp/README.md` for the C++ cross-check.
