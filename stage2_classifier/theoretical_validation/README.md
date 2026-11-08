# p=1 theoretical validation

Classifier v1.0.0-pilot on the completed Stage 1 p=1 grid: N=1000, trial 0, W3=[12000, 14000], 1741 profiles. Labels are rebuilt in memory from the local Stage 1 database. The ODE is not rerun, and the archived thresholds are not changed.

The three checks are separate. A theoretical curve is not treated as the same object as a finite-time label change.

| Check | Boundary | Record |
| --- | --- | --- |
| Eq. (5) | Incoherent-branch onset, \(K=1-\delta\) | [eq5_partial_death](eq5_partial_death/eq5_validation_report.md) |
| Eq. (7) | Small-\(\delta\) Hopf truncation of the incoherent state | [eq7_hopf](eq7_hopf/eq7_validation_report.md) |
| Eq. (8)–(9) | Death equilibrium existence | [death_boundary](death_boundary/death_validation_report.md) |
| Overlay | The three curves on one zoomed phase diagram | [integrated_phase_diagram](integrated_phase_diagram/plot_integrated_phase_diagram.py) |

![Zoomed p=1 phase diagram with Eqs. (5), (7), and (8)–(9)](integrated_phase_diagram/integrated_theoretical_phase_diagram.png)

Eq. (7) is the printed product form, not the additive expansion. The solid orange piece is \(0<\delta\le 0.20\). The dashed orange piece continues only while \(K_H<1-\delta\). Eq. (5) is branch-conditional. Eq. (8)–(9) gives equilibrium existence, not attraction from every initial condition. Its saddle-node and end-point pieces are one boundary: the same teal, solid then dashed, joined at \(\delta_*\approx 0.2956\).

`stage1_runs/` stays local and is not part of this commit. Each script reads that database and writes only its own figure.
