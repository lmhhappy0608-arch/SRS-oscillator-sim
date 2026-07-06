# SRS Oscillator Simulations

Local simulation code for SRS-related oscillator / phase models.

## Main scripts

- `Ariaratnam2001.py` — Ariaratnam-style simulation (meeting requirements).
- `Winfree_phase_3d.py` — Winfree phase 3D analysis and plotting.

## Running

```bash
python3 Ariaratnam2001.py --no-plot
```

Trial outputs are written under `raw_trials/` (excluded from git via `.gitignore`).

## Requirements

Python 3 with NumPy and Matplotlib (and any imports used by the scripts).
