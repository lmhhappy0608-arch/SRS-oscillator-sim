# Eq. (5) validation: low-frequency stopping and Partial Death

Ariaratnam & Strogatz (2001), Eq. (5), gives the coupling at which the lowest-frequency oscillator on the stationary incoherent branch can stop:

\[
K_c(\delta)=1-\delta.
\]

This note compares that branch condition with saved W3 rotation numbers and with Classifier v1 labels. It does not treat \(K=1-\delta\) as a pointwise ground truth for every cell in the phase diagram.

## A. Data and assumptions

The Stage 1 index contains 1741 completed \(p=1\) runs. Every run has \(N=1000\), trial 0, and status `completed`. There is no second trial.

The parameter grid is \(K=0,0.05,\ldots,3\) and \(\delta=0,0.025,\ldots,1\), but it is not a full rectangle. All 41 values of \(\delta\) have the 41 couplings \(K\le 2\). Only \(\delta\in\{0.05,0.20,0.80\}\) also have \(K=2.05,\ldots,3\) (20 further values each). The other 760 cells with \(K>2\) are absent. They are left blank in the figure. They lie well above \(K_c\le 1\) and do not enter the onset comparison.

| Item | Recorded value |
| --- | --- |
| Classifier | Archived `v1.0.0-pilot` in `stage2_classifier/classifier_v1`. Summer `classify_state` is not used. |
| Primary window | W3 \(=[12000,14000]\). The label is the W3 state. |
| Archived near-zero rule | \(\lvert\rho\rvert<0.001\), strict. Death if that fraction equals 1. |
| Partial Death | Longest consecutive near-zero run at least 15, after higher-priority Death and Locking. |
| Structural counts in this note | \(\lvert\rho\rvert<0.01\), strict, on the frequency-sorted profile. This is the threshold requested for the run-length maps. It is wider than the archived classifier tolerance. |
| Solver | Fixed-step RK4, `code/generate_raw_grid.py:rk4_step`. Stored `dt` is 0.1 on all 1741 runs. \(t\) runs from 0 to 14000. The code cap `min(0.1, 2\pi/(20\omega_{\max}))` binds for every \(\delta\in[0,1]\), because \(\omega_{\max}=1+\delta\le 2\) gives a period-based step of at least \(\pi/20\approx 0.157\). |
| Rotation number | \((\theta(t_1)-\theta(t_0))/(t_1-t_0)\) on each stored window. Every file has the window \([12000,14000]\). |
| Frequencies | \(\omega_i=1-\delta+2\delta(i-1)/(N-1)\). `frequency_seed` is null. The maximum absolute deviation from this grid, over all 1741 profiles, is 0. |
| Randomness | `master_seed=0`, recipe `sha256-v1`, mode `paired`, one group `paired-net-0-ic-0`. Network realization id 0, initial-condition id 0. One network seed and one initial-phase seed are shared by all 1741 rows. |
| Initial phases | `numpy` Generator uniform on \([0,2\pi)\), from the stored phase seed. One draw. |
| Network at \(p=1\) | Undirected complete graph, zero diagonal, 499500 upper-triangle edges. Coupling remains \(K/N\), with pulse \(1+\cos\theta\) and phase response \(-\sin\theta\). Regenerating that graph from the shared network seed matches the adjacency hash of one stored run. The other 1740 hashes were not recomputed; they share the same seed and \(p\). |

At \(p=1\) the factor in front of the interaction is \(K/N\), and oscillator \(i\) has no self-pulse. A continuum field normalized by \(N\) rather than by \(N-1\) therefore differs by a relative factor \((N-1)/N=0.999\) when the mean pulse is computed over the other oscillators. That \(1/N\) discrepancy is recorded here and is not removed by shifting \(K_c\).

The derivation of Eq. (5) used below is the one specified for this check: on the stationary incoherent branch the mean-field strength is \(\sigma=K\), a single oscillator obeys \(\dot\theta=\omega-K\sin\theta\), a stable rest equilibrium exists for \(\omega<K\), and \(\omega_{\min}=1-\delta\). The same discrete grid gives a reference for the fifteenth lowest frequency,

\[
K_{15}(\delta)=1-\delta+\frac{28\delta}{999}.
\]

\(K_{15}-K_c=28\delta/999\le 0.028\), which is smaller than the coupling step 0.05. \(K_{15}\) is not an analytic boundary from the paper and is not a prediction of the classifier's actual edge. No onset is interpolated between grid points.

\(\delta=0\) is degenerate: every \(\omega_i=1\), so a run "from the lowest frequency" is only a run in index order. It is excluded from the low-frequency interpretation.

Labels were computed in memory with `classify_window`. The archived classification tables were not rewritten, and the ODE was not rerun.

## B. Numerical observations

W3 state counts on the 1741 profiles: Incoherence 329, Partial Locking 1, Locking 88, Partial Death 348, Death 975.

For every one of the 41 widths, the low-frequency near-zero run (\(\lvert\rho\rvert<0.01\), counted from \(\omega_{\min}\)) is a non-decreasing function of \(K\) on the scanned grid. The run does not shrink and then return. Partial Death labels, where they occur, occupy a single contiguous \(K\) interval and are followed by Death. There is no Partial Death label whose longest \(\lvert\rho\rvert<0.01\) run starts away from \(\omega_{\min}\). All 348 Partial Death profiles have a low-frequency run of at least 15 at both 0.01 and the archived 0.001 tolerance.

Where the profile below \(K_c\) is still incoherent, the low-frequency run turns on at \(K_c\) and then lengthens as \(K\) increases. Two measured examples, both with the 0.01 and 0.001 runs equal:

- \(\delta=0.50\): run 0 at \(K=0.50\), then 60, 133, 221, 326, 454, 611, 807 at \(K=0.55,0.60,\ldots,0.85\), then 1000 (Death) at \(K=0.90\).
- \(\delta=0.80\): one oscillator at \(K=0.20\), then 34 at \(K=0.25\), increasing to 947 at \(K=1.00\), then 1000 at \(K=1.05\).

The bare count of frequencies with \(\omega<K\) is 50 at \((\delta,K)=(0.50,0.55)\) and 32 at \((0.80,0.25)\). The measured runs, 60 and 34, are close to those counts at the first grid point. Deeper in the same columns the measured runs are larger: 611 against about 300 at \((0.50,0.80)\), and 947 against 500 at \((0.80,1.00)\). Eq. (5) fixes the start of stopping on the incoherent branch. It does not by itself set the width of the stopped set after that branch has changed.

No Incoherence cell with \(K>1-\delta\) has a zero low-frequency run. The incoherent cells that sit exactly on a grid value \(K=1-\delta\) and still have a zero run are the marginal points of the strict condition \(\omega<K\).

The complementary small-\(\delta\) region does not show a low-frequency edge. For \(0\le\delta\le 0.175\) the W3 sequence is Incoherence, then Locking, then Death from \(K=0.80\) through the top of the scan. At that Death boundary all 1000 rotation numbers are near zero. For \(0.025\le\delta\le 0.175\) there are 16 such Death cells with \(K<1-\delta\). None of them is a short low-frequency run.

## C. Comparison with Eq. (5)

Three locations are kept separate.

- **Theoretical onset:** \(K_c=1-\delta\), the value where \(\omega_{\min}=K\). The strict rest condition is \(K>\omega_{\min}\).
- **Numerical near-zero onset:** the smallest scanned \(K\) at which the low-frequency \(\lvert\rho\rvert<0.01\) run is at least 1, or at least 15. If it never occurs, the onset is not observed.
- **Classifier Partial Death onset:** the smallest scanned \(K\) with W3 state Partial Death. Death has priority, so a jump straight to Death does not produce this onset.

The Eq. (5) comparison set is the 30 widths \(\delta=0.25,0.275,\ldots,0.975\). On each of these, every saved cell with \(K<1-\delta\) is Incoherence, so the label immediately below the line has not already moved to Locking or Death. This is an operational screen on the saved attractor. It is not a demonstration that the trajectory is the paper's stationary incoherent solution.

On that set the grid resolves the onsets as follows.

- The first low-frequency run of length at least 1 is the first grid point strictly above \(1-\delta\) for 26 widths. For \(\delta\in\{0.80,0.85,0.90,0.95\}\) it is the grid point \(K=1-\delta\) itself, and the run length there is 1. That single oscillator satisfies \(0.001\le\lvert\rho\rvert<0.01\), so the archived 0.001 rule still counts a run of 0 and the state remains Incoherence.
- The first run of length at least 15 is the first grid point strictly above \(K_{15}\) for 29 of the 30 widths. The exception is \(\delta=0.925\): \(K_{15}=0.1009\), and at \(K=0.10\) the 0.01 run is already 15 while the 0.001 run is 14.
- The first Partial Death label is the first grid point strictly above \(K_{15}\) for all 30 widths. Signed gaps from the observed label to \(1-\delta\) are one grid step, from \(+0.025\) to \(+0.075\). Those gaps are the 15-oscillator offset plus the 0.05 spacing. They are not interpolated critical values.

Outside this set the same three onsets are not the same object. For \(\delta\le 0.175\), and at \(\delta=0.20\) and \(0.225\), Partial Death is not observed: the first cell with a long near-zero run is already Death.

| \(\delta\) | \(1-\delta\) | \(K_{15}\) | First \(K\), run \(\ge 1\) | First \(K\), run \(\ge 15\) | First Partial Death | What the cells show |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.950 | 0.951 | 0.80 (1000, Death) | 0.80 | not observed | Locking for \(K=0.15\)–\(0.75\), then full death. \(K_c\) is above that death boundary. |
| 0.20 | 0.800 | 0.806 | 0.80 (1000, Death) | 0.80 | not observed | Locking only at \(K=0.65\); run jumps from 0 at \(K=0.75\) to 1000 at \(K=0.80\). |
| 0.25 | 0.750 | 0.757 | 0.80 | 0.80 | 0.80 | First grid point above both lines. Run at \(K=0.75\) is 0. |
| 0.50 | 0.500 | 0.514 | 0.55 (60) | 0.55 | 0.55 | Same grid point. Run at \(K=0.50\) is 0, then the run lengthens. |
| 0.80 | 0.200 | 0.222 | 0.20 (1, still Incoherence) | 0.25 (34) | 0.25 | Marginal oscillator on \(K=1-\delta\). The label starts on the first point above \(K_{15}\). |
| 0.925 | 0.075 | 0.101 | 0.10 (15 at 0.01; 14 at 0.001) | 0.10 | 0.15 | The archived tolerance holds the label for one grid step. |
| 1.00 | 0.000 | 0.028 | 0.00 (5) | 0.05 (27) | 0.05 | At \(K=0\), \(\rho=\omega\) already puts five low frequencies under 0.01. |

## D. Disagreements and interpretation

**Narrow frequency spread, \(\delta\le 0.175\).** At \(\delta=0.05\), Locking occupies \(K=0.15\) through \(0.75\) with a profile range of order \(10^{-4}\) or smaller and a low-frequency run of 0. Death begins at \(K=0.80\), with all 1000 oscillators near zero, while \(K_c=0.95\). The same death column \(K=0.80\) appears for every \(\delta\le 0.175\). Direct evidence: the state sequence, the run length 1000, and \(K_c>0.80\). The incoherent-branch hypothesis of Eq. (5) is already false on the saved profiles before \(K\) reaches \(1-\delta\). These cells are a different attractor, reached through locking. They are not counted as an onset error for Eq. (5).

**\(\delta=0.20\): a one-cell locking pocket, then a jump to full death.** The state is Incoherence on \(K\le 0.60\), Locking at the single point \(K=0.65\) (range \(1\times 10^{-4}\)), Incoherence again at \(K=0.70\) (range 0.0157) and \(K=0.75\) (range 0.1939, run 0), and Death at \(K=0.80\) (run 1000). \(K_c=0.80\). Partial Death is not observed. Direct evidence is that sequence. The grid cannot show whether a partial-death interval shorter than 0.05 exists between \(K=0.75\) and \(K=0.80\). With one initial condition, the return from Locking to Incoherence cannot be separated from a peculiarity of that draw. Either reading is enough to avoid calling \(K=0.80\) a unique phase boundary.

**\(\delta=0.225\): the only Partial Locking cell.** At \(K=0.75\), just below \(K_c=0.775\), the low-frequency run is 0, the profile range is 0.768, and the archived rule returns Partial Locking. At \(K=0.80\) the profile is Death. Direct evidence: no near-zero oscillators are present in that Partial Locking cell, so the label is not a low-frequency death structure that the classifier missed. The Eq. (5) onset, if it occurs, lies inside the unresolved gap \((0.75,0.80]\).

**One tolerance cell, \(\delta=0.925\), \(K=0.10\).** The low-frequency run is 15 for \(\lvert\rho\rvert<0.01\) and 14 for \(\lvert\rho\rvert<0.001\). Classifier v1 therefore returns Incoherence; Partial Death starts at \(K=0.15\), where both runs are 43. This is the only \(p=1\) cell in which a 0.01-run of at least 15 fails the archived 0.001 run length without already being Death or Locking. Direct evidence: the two masks on that profile. \(K_{15}=0.1009\) lies 0.0009 above the grid point \(K=0.10\), so the looser mask crossing 15 there is also consistent with a threshold sitting just above a grid line. The archived rule is doing what it states. The task description's 0.01 cut is not the archived cut; on this database the two cuts disagree by one oscillator in this cell and agree on the Partial Death decision everywhere else in the clean set.

**Stopped-set size above onset.** At the first Partial Death cell the run is near the count of oscillators with \(\omega<K\). Farther above \(K_c\) it is substantially larger, as in 611 versus about 300 at \(\delta=0.50\), \(K=0.80\). Direct evidence is that pair of counts. A larger mean field after some oscillators have stopped would recruit more of the low-frequency end; this dataset does not contain a measured \(\sigma\), so that mechanism stays an interpretation. It is independent of whether the onset sits on Eq. (5).

**Marginal grid points.** At \(\delta\in\{0.80,0.85,0.90,0.95\}\) the lowest oscillator has \(0.001\le\lvert\rho\rvert<0.01\) exactly at \(K=1-\delta\), where \(\omega_{\min}=K\). Direct evidence: run length 1 at tolerance 0.01 and 0 at tolerance 0.001, with the state still Incoherence. A finite-time rotation can be small at the saddle-node \(\omega=K\) without meeting the archived rest cut. The time series was not re-inspected; only the stored W3 number is known.

**\(\delta=1\).** \(\omega_{\min}=0\). At \(K=0\) the low-frequency run is 5 for \(\lvert\rho\rvert<0.01\) and 1 for \(\lvert\rho\rvert<0.001\), and the state is Incoherence. Those counts match \(\rho=\omega\) for the free oscillators with \(\omega<0.01\) and \(\omega<0.001\). They are not coupling-induced death. Partial Death at \(K=0.05\) is the first grid point above \(K_{15}=0.028\).

## E. Conclusions

1. On the 30 widths where every saved profile below \(K=1-\delta\) is still Incoherence, the low-frequency near-zero run appears at \(K=1-\delta\) or at the next grid point, and it lengthens as \(K\) increases. That is the onset mechanism described by Eq. (5), seen at a coupling resolution of 0.05. The same line does not organize the small-\(\delta\) side of the diagram, where the saved attractor locks and then dies while \(K\) is still below \(1-\delta\).

2. Classifier v1 tracks that low-frequency structure under its own rules. Every Partial Death label is a low-frequency run of at least 15, including at the archived tolerance 0.001. None is a near-zero band in the middle of the frequency range. Death replaces Partial Death when the near-zero fraction reaches 1.

3. The visible offset between \(K=1-\delta\) and the Partial Death boundary is the fifteen-oscillator requirement, the 0.05 grid, and one cell in which 14 rather than 15 oscillators meet \(\lvert\rho\rvert<0.001\). On the 30-width set the Partial Death onset is the first scanned \(K\) above \(K_{15}\). A separate, larger discrepancy is the width of the stopped set well above onset, which exceeds the count of frequencies with \(\omega<K\).

4. The agreement is independent evidence for the low-frequency-death content of Classifier v1: the thresholds were not fit to Eq. (5), and the comparison uses the new \(p=1\) grid rather than the 732-run archive on which the rules were chosen. It supports the onset location in the incoherent region. It does not certify Locking, Partial Locking, or the Method C penalties, and a matching label is not a proof that the finite-\(N\) attractor is the paper's incoherent branch.

5. The check uses one initial condition. At \(p=1\) the graph itself has no ensemble, but the phases do, and that variance is unmeasured. \(N=1000\), the \(K/N\) normalization, and the missing self-pulse keep the runs a finite-\(N\) distance from the continuum limit. The coupling grid cannot locate an onset more finely than 0.05, and \(K>2\) is missing for 38 of 41 widths. Eq. (5) remains a statement about one solution branch.

The figure is `eq5_validation_figure.png`. Gray cells are missing runs. The solid curve is Eq. (5). The dashed curve is the discrete \(K_{15}\) reference.
