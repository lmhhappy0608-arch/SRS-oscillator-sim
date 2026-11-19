# Full-grid annealed boundaries on Classifier v2.0.0

N=1000, W3=[12000, 14000], α=2, β=1, one trial. The classifier table is read only. No label, rule, database row, or archived figure was changed. Missing Hopf values are blank. No curve is interpolated or extrapolated across a gap.

The phase-diagram grid, read from `classification_results.csv`, is δ = 0, 0.025, …, 1 (41 values) and K = 0, 0.05, …, 3. The rectangle is incomplete: many high-K cells are absent and stay masked.

## What was computed

For every grid δ, and for p = 1, 0.8, 0.4, 0.2:

- **All-to-all Hopf.** One root of Ariaratnam & Strogatz Eq. (6) on the all-drifting branch, the same solver checked against the Step 1 linear response. Eq. (7) is stored only for 0 < δ ≤ 0.20 and is not drawn.
- **Naive pK Hopf.** K_H(p=1)/p, filled only when that Eq. (6) root is valid.
- **Degree-based annealed Hopf.** An independent root of D(λ; K, p, δ, N) = 0, using the Step 2 solver (ε → 0+ extrapolation, degree-tail check, Fourier check). A successful residual is not enough: the root is kept only when every retained degree class is all-drifting and the continuation converges.
- **All-drifting edge.** K = (1−δ)/p. Homogeneous annealed reference.
- **Death existence.** Eq. (8)–(9), then divided by p. Saddle-node and end-point are separate mechanisms. Existence is not attraction and is not the classifier Death label.

## 1. Where a valid Hopf boundary was obtained

On this grid the all-drifting Hopf root exists at

δ = 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200

for every computed p. All 32 degree-based solves (8 widths × 4 probabilities) returned `valid_all_drifting` / `converged`. Residuals are about 10^−16 to 10^−13. The binomial tail kept out of the sum is about 4.6×10^−18. The largest pinned-neighbor mass on a kept root is 4.2×10^−10, at p=0.2, δ=0.20, below the 10^−8 blanking threshold.

Recovery against the archived Step 1 and Step 2 tables passed. At δ = 0.05, 0.10, 0.20 the new Eq. (6) values match Step 1, and the new degree-based values match Step 2.

δ = 0 is outside the validity range: the frequency support has zero width and the Eq. (6) integral is undefined. Hopf fields are blank. The edge and the death-existence values are still recorded.

## 2. Where the Hopf branch stops being available

The next grid point, δ = 0.225, and every larger grid δ through 1, is `no_root_detected`: the Eq. (6) scan found no sign-changing root with 0 < K < 1−δ. That is a failed search on the studied branch, not a proof that no Hopf eigenvalue exists anywhere.

An 8-step bisection, stored only in the summary JSON and not drawn, places the last detected Eq. (6) root at δ = 0.22275 (K = 0.71840, Ω = 0.61829) and the first miss at δ = 0.22285. At that last root, K is still below 1−δ = 0.777, so the branch ends inside the all-drifting interval rather than by running into the pinning edge. The degree-based solver was not continued off the grid. The drawn Hopf curves stop at the last valid grid point, δ = 0.20, and are not joined to δ = 0.225.

Eq. (7) at δ = 0.20 is 0.55435, about 2.6% below the Eq. (6) root 0.56905. It was not extended past δ = 0.20.

The mixed pinned/drifting branch was not analysed. It needs its own stationary state and its own linearisation. The present susceptibility was not reused there.

## 3. Degree-based theory versus naive pK scaling

Both methods are valid at the same eight widths. The relative deviation (K_degree − K_scaling) / K_scaling stays within a third of a percent:

| p | max \|relative deviation\| | at δ | sign |
| --- | ---: | ---: | --- |
| 0.2 | 0.299% | 0.025 | degree below scaling |
| 0.4 | 0.050% | 0.025 | degree below scaling |
| 0.8 | 0.086% | 0.200 | degree above scaling |

Most of that gap is the finite-N mean-neighbor-degree factor (2p−1)/(1+(N−2)p), the dotted lines on the deviation figure. The extra spread from the degree distribution is smaller and grows as δ increases and p decreases. Its largest value on this grid is the Step 2 heterogeneity correction, 0.181%, at p=0.2, δ=0.20. On the phase diagrams the two Hopf curves lie on top of each other; the deviation figure is the comparison.

## 4. Which classifier regions the Hopf curve crosses

The curve is the degree-based root. Placement is relative to the saved cells at that δ. It is not a claim that the classifier boundary is the Hopf boundary.

**p = 0.2.** At δ = 0.025 the root (K = 0.318) sits between Partial Locking at K = 0.30 and Locking at K = 0.35. Through δ = 0.075 it stays in the Partial Locking band. From δ = 0.10 to 0.15 it lies on the Incoherence / Partial Locking boundary. At δ = 0.175 (K = 2.386) it is above every saved cell in that column; those cells stop in Incoherence and the point is drawn on a missing cell. At δ = 0.20 (K = 2.842) it is between two Incoherence cells.

**p = 0.4.** At δ = 0.025 and 0.05 the root is between Partial Locking and Locking. From δ = 0.075 to 0.125 it is inside Partial Locking. At δ = 0.15 and 0.20 it is inside Incoherence. At δ = 0.175 it is between Incoherence (K = 1.15) and Partial Locking (K = 1.20).

**p = 0.8.** At δ = 0.025 (K = 0.080) the root is between Incoherence at K = 0.05 and Locking at K = 0.10. From δ = 0.05 to 0.15 it lies between Partial Locking and Locking. At δ = 0.175 it is inside Partial Locking. At δ = 0.20 (K = 0.712) it is between Incoherence at K = 0.70 and Partial Locking at K = 0.75.

On all three diagrams the Hopf curve stays in Locking, Partial Locking, Incoherence, or a missing cell. It does not enter Partial Death or Death.

## 5. Which boundaries sit near a classifier partition

**All-drifting edge.** Once K = (1−δ)/p is inside the scan, it follows the staircase between Incoherence and Partial Death. For p = 0.8 it is inside from δ = 0 (K = 1.25). For p = 0.4 it enters near the top of the scan. For p = 0.2 it enters only at δ = 0.4, where (1−δ)/p drops through K = 3. The staircase is close to this homogeneous edge and is not close to the Hopf curve. At large δ the classifier leaves Incoherence near this edge, not near a Hopf root.

**Death existence, scaled by 1/p.** The mechanism changes at δ_* = 0.2956, where the p = 1 existence value is K = 0.814. Grid rows on either side keep their own mechanism; the figure joins the two pieces by evaluating that one point, not by interpolating K.

- p = 0.8: the curve starts at K = 0.962, near the Locking / Death contact around K = 1, then rises through Partial Death to K = 1.40 at δ = 1. The thick Death band above K ≈ 1 is much wider than this curve. The curve does not trace the Death / Partial Death interface.
- p = 0.4: the curve starts at K = 1.925, near the top of Locking and the start of Partial Death, then climbs through Partial Death and the missing cells to K = 2.80 at δ = 1. The isolated Death columns are not followed.
- p = 0.2: the scaled existence value is at least K = 3.85, above the scan, so it is absent from the figure. The saved p = 0.2 table has no Death label. Partial Death at large δ lies far below this curve.

**Hopf versus Locking.** At the smallest δ the Hopf root passes near the first Locking cells. By δ = 0.20, Locking and Partial Locking extend well above the Hopf root, and the root itself sits in or beside Incoherence. The two boundaries agree only in a narrow corner of the diagram.

## 6. Differences that compare different objects

These mismatches are expected if the objects differ, and they are not by themselves an error in the root or in the classifier.

- The Hopf root is the linear onset of a collective pulse on the annealed all-drifting incoherent state.
- Locking and Partial Locking record rotation-number structure on one quenched finite-N trajectory.
- The black edge is the homogeneous condition pK = 1−δ. It is not a degree-resolved pinning boundary and not a Hopf condition.
- The brown curve is homogeneous death *existence*, divided by p. A state can exist without attracting the sampled initial condition, and the classifier Death label is an attraction label on a finite quenched network.
- Degree heterogeneity at N = 1000 moves the annealed Hopf root by less than 0.3%. It does not account for the gap between the Hopf curve and the Incoherence / Partial Death staircase.

## Files

- `phase_diagram_theory_p_0.20.png`, `_0.40.png`, `_0.80.png`
- `hopf_degree_vs_scaling_deviation.png`
- `full_grid_boundaries.csv` (164 rows: 4 probabilities × 41 widths)
- `full_grid_boundaries_summary.json` (endpoint bracket, recovery, cell placement)

Eq. (7) is in the CSV as `K_eq7_p1` and `K_eq7_scaling` and is omitted from the figures. Invalid Hopf entries are empty, with the reason in `notes`.

## p = 0.6

Classifier v2.0.0 has no p=0.6 cells, so the first `phase_diagram_theory_p_0.60.png` used the same δ step 0.025 and K step 0.05, with a gray background. The curves are the same annealed boundaries as the other panels. Degree-based roots again exist only for δ = 0.025 through 0.200. They recover the three Step 2 values at δ = 0.05, 0.10, and 0.20. The largest relative deviation from naive pK scaling on this grid is 0.064% at δ = 0.20. The deviation figure includes this probability.

## Extended Stage 1 cells

After the Stage 1 scan finished, 3263 completed cells were absent from the archived Classifier v2.0.0 table: all 1741 p=0.6 cells, 760 new p=0.4 cells with K>2, and 762 p=0.2 cells (K>2, plus the previously unfinished K=1.95 and K=2 at δ=1). Those cells were labeled with the unchanged v2.0.0 rule and written to `extended_classification.csv`. The archived classification table and its figures were not edited. The three overlay figures were redrawn from the archived labels plus these new labels. p=0.6 follows the saved p=0.8 coverage, so K>2 remains gray except at δ=0.05, 0.20, and 0.80. On the new p=0.2 cells the labels are Locking, Partial Locking, Incoherence, and Partial Death; Death does not appear. On the new p=0.4 cells the labels are Death and Partial Death.
