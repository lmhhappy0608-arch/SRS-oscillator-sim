# Eq. (5) on ER networks: annealed scaling check

## 1. Research question

Ariaratnam & Strogatz (2001), Eq. (5), gives the coupling at which the slowest oscillator on the stationary incoherent branch can stop rotating, \(K_c(1,\delta)=1-\delta\), for the all-to-all Winfree model. This note asks whether the completed Stage 1 Erdős–Rényi scans are consistent with the annealed mean-field scaling

\[pK_c(p,\delta)\approx 1-\delta,\qquad K_c(p,\delta)\approx\frac{1-\delta}{p}.\]

That relation is a hypothesis about one solution branch. It is not an exact critical formula for a finite ER graph, and it is not a boundary between every pair of classifier states. Eq. (7) and Eqs. (8)–(9) are not tested here. The previous p=1 check is reused as the baseline and is not refit.

## 2. Model and mean-field scaling assumption

The saved trajectories integrate

\[\dot\theta_i=\omega_i-\frac{K}{N}\sin\theta_i\sum_{j=1}^{N}A_{ij}(1+\cos\theta_j),\]

with \(N=1000\), an undirected ER adjacency matrix, a zero diagonal, and \(\omega_i=1-\delta+2\delta(i-1)/(N-1)\). The sum is normalized by \(N\), not by \(N-1\) or by the realized degree. Replacing \(A_{ij}\) by \(p\) for \(i\neq j\) makes the expected interaction strength \(pK(N-1)/N\). The boxed comparison drops the factor \((N-1)/N\) and tests \(pK\) against \(1-\delta\). A one-line finite-\(N\) reference that restores only the expected mean degree is

\[K_{c,\mathrm{finite}N}(p,\delta)=\frac{1-\delta}{p(1-1/N)}.\]

The two predictions differ by about \(K_c/999\), at most a few thousandths on this grid. The coupling step is 0.05, so the scan cannot separate them. \(K_{c,\mathrm{finite}N}\) does not include degree fluctuations and is not a solved finite-graph threshold. The boundary rule used below is the strict rest condition \(pK>1-\delta\). When \((1-\delta)/p\) lands on a scanned \(K\), that point is the equality \(K=K_{c,\mathrm{MF}}\) and is reported as such, with distance zero. The first grid point inside the strict region is the next scanned \(K\). No onset is interpolated.

Near-zero means \(|\rho|<0.001\) on W3 \(=[12000,14000]\). The raw Partial Death criterion is a run of at least 15 consecutive near-zero oscillators in frequency order. The classifier label Partial Death is that run after the archived priority Death, then Locking, then Partial Death. A higher-priority label does not erase the raw run. W1 \(=[8000,10000]\) and W2 \(=[10000,12000]\) are diagnostics. They do not assign the state.

## 3. Dataset audit

The Stage 1 index `/Users/mohan/Desktop/SRS/helper/stage1_runs/index.sqlite` has 6964 rows and 6962 completed runs. Status counts: completed 6962, pending 1, running 1. This is the full saved grid. It is not the 732-run pilot archive, and it is not 7324 runs.

Incomplete rows, excluded from every onset: `bc7f4faf661a90d2f2b43455b7538fcd` (running, p=0.2, δ=1, K=1.95); `6bfc891de871edc9c0b416ec405f843e` (pending, p=0.2, δ=1, K=2).

Every row has N=1000, solver count 1, dt=0.1, t from 0.0 to 14000.0, and 0 non-null frequency seeds. Distinct trials: 1. Distinct network seeds: 1. Distinct initial-phase seeds: 1. Randomness modes: 1. Master seeds: 1. Seed recipes: 1.

| p | δ columns | K counts | Upper end | Interior holes |
| --- | --- | --- | --- | --- |
| 0.2 | 41 | 1 cols × 39 K, 37 cols × 41 K, 3 cols × 61 K | Kmax=1.9: 1, Kmax=2: 37, Kmax=3: 3 | 0 |
| 0.4 | 41 | 38 cols × 41 K, 3 cols × 61 K | Kmax=2: 38, Kmax=3: 3 | 0 |
| 0.8 | 41 | 38 cols × 41 K, 3 cols × 61 K | Kmax=2: 38, Kmax=3: 3 | 0 |
| 1.0 | 41 | 38 cols × 41 K, 3 cols × 61 K | Kmax=2: 38, Kmax=3: 3 | 0 |

δ runs from 0 to 1 in steps of 0.025 (41 values). K runs from 0 in steps of 0.05. For every p, 38 widths stop at K=2 and the three widths δ∈{0.05, 0.20, 0.80} continue to K=3. The p=0.2, δ=1 column stops at K=1.90 because K=1.95 is still marked running and K=2.00 is still pending. No other interior K is missing. There is no p=0.05 and no p=0.60 in the index. The maximum absolute deviation of any saved ω from the deterministic grid is 0.000e+00. Frequency order matches oscillator index on 6962 of 6962 profiles. At K=0 the maximum |ρ−ω| on W3 is 1.819e-11.

One network seed and one initial-phase seed are shared by all four values of p. Rebuilding the undirected ER graphs with `stage1.model.make_adjacency` and seed 8250318217634423533 matched the stored `adjacency_sha256` on 6962 metadata files (0 mismatches, 0 missing JSON files). The four graphs are nested: True. Degree in the sections below is this rebuilt degree, not a new random draw.

| p | Edges | Mean degree | Std | Min | Max | Degree of ω index 0 | Simple graph |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 99271 | 198.54 | 12.54 | 163 | 239 | 191 | yes |
| 0.4 | 199170 | 398.34 | 15.42 | 348 | 443 | 418 | yes |
| 0.8 | 399066 | 798.13 | 12.29 | 760 | 846 | 800 | yes |
| 1.0 | 499500 | 999.00 | 0.00 | 999 | 999 | 999 | yes |

Classifier v1.0.0-pilot, config SHA-256 `5e77fae526cd6060b9f097db70d327f241c75a4579d7e9606c2070f360d1be44` (matches the archive manifest). `classify.py` SHA-256 `2d3488de136f0420cc6d570f6c23640b7759b4f5c728957c2b0da4d2b517d58b` does not equal the manifest checksum `b8dcc227ced4404952a1e93776abcf51ecd8bc77ad73e9db984fa477eb907bad`; the worktree difference from git HEAD is in the phase-diagram drawing helpers. State decisions were made by the archived `classify_window` rules: W3 primary, |ρ|<0.001, death fraction 1, locking range 0.01, partial-death run 15, priority Death > Locking > Partial Death > Partial Locking > Incoherence. Method C ran on 2916 profiles. The other 4046 profiles were already Death, Locking, or Partial Death before Method C; a direct `classify_window` check of 117 fast-path or boundary profiles found 0 disagreements. Git HEAD `910f9c2700396a0cc081f0df5c96089b84bb8b2c`.

## 4. p=1 baseline reproduction

The previous Eq. (5) note used the same p=1 grid, the same W3 window, and the same archived classifier. Its structural run lengths were counted at |ρ|<0.01. The ER comparison below uses the archived cut |ρ|<0.001. Both cuts are recomputed here. A mismatch at 0.01 would be a data or window problem. A mismatch that appears only at 0.001 is the tolerance the previous note already described.

| Result | Check | Detail |
| --- | --- | --- |
| PASS | p=1 completed profiles | found 1741 |
| PASS | p=1 Incoherence count | found 329, previous report 329 |
| PASS | p=1 Partial Locking count | found 1, previous report 1 |
| PASS | p=1 Locking count | found 88, previous report 88 |
| PASS | p=1 Partial Death count | found 348, previous report 348 |
| PASS | p=1 Death count | found 975, previous report 975 |
| PASS | 30 widths δ=0.25…0.975 are clean | 30 of 30 |
| PASS | four marginal δ at K=1-δ with a 0.01 run of 1 | 4 |
| PASS | other 26 widths onset at the first K above 1-δ | 26 |
| PASS | Partial Death is the first K above K15 on all 30 widths | 30 |
| PASS | Partial Death gaps from 1-δ span 0.025 to 0.075 | min 0.02499999999999991, max 0.07500000000000004 |
| PASS | example δ=0.500, K=0.50 | prefix_01=0, prefix_001=0 |
| PASS | example δ=0.500, K=0.55 | prefix_01=60, prefix_001=60 |
| PASS | example δ=0.800, K=0.20 | prefix_01=1, prefix_001=0 |
| PASS | example δ=0.925, K=0.10 | prefix_01=15, prefix_001=14 |
| PASS | δ=0.925 Partial Death starts at K=0.15 | K=0.15 |

Every baseline check matches the previous p=1 note: the same 1741 profiles, the same five state counts, the same 30 clean widths δ=0.25, 0.275, …, 0.975, the same 0.01-run onset rule (26 widths on the first K above 1−δ, and δ∈{0.80, 0.85, 0.90, 0.95} on the equality point with a single oscillator inside the 0.01 cut but outside 0.001), and Partial Death on the first grid point above K_15=1−δ+28δ/999. The signed Partial Death gaps from 1−δ run from 0.025 to 0.075. δ=0, the small-δ locking columns, δ=0.20, and δ=0.225 remain outside that clean set, as before.

## 5. Applicability screening

For each (p, δ) the completed K values are read in increasing order. A column is clean when every saved cell with K below the annealed prediction is Incoherence, and every saved cell before the first near-zero oscillator is also Incoherence. That is the operational incoherent branch. It is not a proof that the finite-time trajectory is the paper's stationary incoherent solution. A column is competing when Locking or Partial Locking, or any other non-Incoherence state, occurs before the prediction. It is interrupted when the first departure from Incoherence is Locking or Partial Locking after the prediction but before any near-zero oscillator. Out-of-range means the prediction lies above the largest scanned K. Those columns are censored. They are not counted as disagreements. Insufficient coverage would mean a missing K within one step of the prediction. Upper edge means the prediction equals the largest scanned K, so the strict region K>K_c was not scanned, and no near-zero oscillator was seen. δ=0 is degenerate because every natural frequency equals 1. A non-monotonic sequence is one in which a state returns after a different state. Those sequences are kept.

| p | Clean | Competing | Interrupted | Out of range | Upper edge | Insufficient | Unresolved | Degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 17 | 0 | 0 | 24 | 0 | 0 | 0 | 0 |
| 0.4 | 32 | 2 | 0 | 7 | 0 | 0 | 0 | 0 |
| 0.8 | 27 | 14 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.0 | 31 | 10 | 0 | 0 | 0 | 0 | 0 | 0 |

Coverage of the annealed prediction, before the state screen: p=1 is inside the scan for all 41 widths; p=0.8 is inside for all 41; p=0.4 is outside for δ=0, 0.025, 0.075, 0.10, 0.125, 0.15, and 0.175 (K_c>2 while the scan ends at 2), and inside otherwise, including the K≤3 strips at δ=0.05 and 0.20; p=0.2 is outside for δ≤0.575 (K_c≥2.125, above K=2, and also above K=3 on the three extended strips). At p=0.2, δ=0.60, K_c=2 equals K_max, so there is no scanned point with K>K_c.

Columns that cannot be read as an incoherent-branch test of Eq. (5). `First event` is the first saved state that is not Incoherence.

| p | δ | K_c,MF | Class | State returns | First event | Sequence |
| --- | --- | --- | --- | --- | --- | --- |
| 0.4 | 0.050 | 2.375 | competing | yes | Partial Locking @ 0.25 | Inc[0.00,0.20] → PL[0.25,0.30] → Lock[0.35,1.85] → PL@1.90 → Death[1.95,3.00] |
| 0.4 | 0.200 | 2.000 | competing | no | Partial Locking @ 1.50 | Inc[0.00,1.45] → PL[1.50,1.95] → PD@2.00 → Death[2.05,3.00] |
| 0.8 | 0.000 | 1.250 | competing | no | Locking @ 0.00 | Lock[0.00,0.95] → Death[1.00,2.00] |
| 0.8 | 0.025 | 1.219 | competing | no | Locking @ 0.10 | Inc[0.00,0.05] → Lock[0.10,0.95] → Death[1.00,2.00] |
| 0.8 | 0.050 | 1.187 | competing | no | Partial Locking @ 0.15 | Inc[0.00,0.10] → PL@0.15 → Lock[0.20,0.95] → Death[1.00,3.00] |
| 0.8 | 0.075 | 1.156 | competing | no | Partial Locking @ 0.20 | Inc[0.00,0.15] → PL@0.20 → Lock[0.25,0.95] → Death[1.00,2.00] |
| 0.8 | 0.100 | 1.125 | competing | no | Partial Locking @ 0.25 | Inc[0.00,0.20] → PL[0.25,0.30] → Lock[0.35,0.95] → Death[1.00,2.00] |
| 0.8 | 0.125 | 1.094 | competing | no | Partial Locking @ 0.35 | Inc[0.00,0.30] → PL[0.35,0.40] → Lock[0.45,0.95] → Death[1.00,2.00] |
| 0.8 | 0.150 | 1.062 | competing | yes | Partial Locking @ 0.40 | Inc[0.00,0.35] → PL@0.40 → Inc@0.45 → PL@0.50 → Lock[0.55,0.90] → PL@0.95 → Death[1.00,2.00] |
| 0.8 | 0.175 | 1.031 | competing | yes | Partial Locking @ 0.45 | Inc[0.00,0.40] → PL@0.45 → Inc@0.50 → PL[0.55,0.65] → Lock[0.70,0.85] → PL[0.90,0.95] → Death[1.00,2.00] |
| 0.8 | 0.200 | 1.000 | competing | yes | Partial Locking @ 0.55 | Inc[0.00,0.50] → PL@0.55 → Inc[0.60,0.70] → PL[0.75,0.85] → Inc@0.90 → PL@0.95 → Death[1.00,3.00] |
| 0.8 | 0.225 | 0.969 | competing | yes | Partial Locking @ 0.55 | Inc[0.00,0.50] → PL@0.55 → Inc[0.60,0.75] → PL@0.80 → Inc[0.85,0.95] → PD@1.00 → Death[1.05,2.00] |
| 0.8 | 0.250 | 0.938 | competing | yes | Partial Locking @ 0.60 | Inc[0.00,0.55] → PL@0.60 → Inc[0.65,0.95] → PD@1.00 → Death[1.05,2.00] |
| 0.8 | 0.275 | 0.906 | competing | yes | Partial Locking @ 0.60 | Inc[0.00,0.55] → PL@0.60 → Inc[0.65,0.90] → PD[0.95,1.00] → Death[1.05,2.00] |
| 0.8 | 0.300 | 0.875 | competing | yes | Partial Locking @ 0.65 | Inc[0.00,0.60] → PL@0.65 → Inc[0.70,0.90] → PD[0.95,1.00] → Death[1.05,2.00] |
| 0.8 | 0.400 | 0.750 | competing | yes | Partial Locking @ 0.65 | Inc[0.00,0.60] → PL@0.65 → Inc[0.70,0.75] → PD[0.80,1.05] → Death[1.10,2.00] |
| 1.0 | 0.000 | 1.000 | competing | no | Locking @ 0.00 | Lock[0.00,0.75] → Death[0.80,2.00] |
| 1.0 | 0.025 | 0.975 | competing | no | Locking @ 0.10 | Inc[0.00,0.05] → Lock[0.10,0.75] → Death[0.80,2.00] |
| 1.0 | 0.050 | 0.950 | competing | no | Locking @ 0.15 | Inc[0.00,0.10] → Lock[0.15,0.75] → Death[0.80,3.00] |
| 1.0 | 0.075 | 0.925 | competing | no | Locking @ 0.20 | Inc[0.00,0.15] → Lock[0.20,0.75] → Death[0.80,2.00] |
| 1.0 | 0.100 | 0.900 | competing | no | Locking @ 0.30 | Inc[0.00,0.25] → Lock[0.30,0.75] → Death[0.80,2.00] |
| 1.0 | 0.125 | 0.875 | competing | no | Locking @ 0.35 | Inc[0.00,0.30] → Lock[0.35,0.75] → Death[0.80,2.00] |
| 1.0 | 0.150 | 0.850 | competing | no | Locking @ 0.45 | Inc[0.00,0.40] → Lock[0.45,0.75] → Death[0.80,2.00] |
| 1.0 | 0.175 | 0.825 | competing | no | Locking @ 0.50 | Inc[0.00,0.45] → Lock[0.50,0.75] → Death[0.80,2.00] |
| 1.0 | 0.200 | 0.800 | competing | yes | Locking @ 0.65 | Inc[0.00,0.60] → Lock@0.65 → Inc[0.70,0.75] → Death[0.80,3.00] |
| 1.0 | 0.225 | 0.775 | competing | no | Partial Locking @ 0.75 | Inc[0.00,0.70] → PL@0.75 → Death[0.80,2.00] |

18 columns have a state that returns later in the K scan: 10 of them are the competing rows marked above, and 8 are out of range. No clean column returns to an earlier state. The out-of-range returns are kept here and are not scored.

| p | δ | Class | Sequence |
| --- | --- | --- | --- |
| 0.2 | 0.075 | out_of_range | Inc[0.00,0.75] → PL[0.80,1.00] → Inc@1.05 → PL@1.10 → Lock[1.15,2.00] |
| 0.2 | 0.100 | out_of_range | Inc[0.00,0.70] → PL@0.75 → Inc[0.80,1.00] → PL[1.05,1.20] → Inc@1.25 → PL[1.30,1.45] → Inc@1.50 → PL@1.55 → Lock[1.60,2.00] |
| 0.2 | 0.125 | out_of_range | Inc[0.00,1.20] → PL[1.25,1.35] → Inc[1.40,1.60] → PL[1.65,2.00] |
| 0.4 | 0.075 | out_of_range | Inc[0.00,0.30] → PL[0.35,0.50] → Lock[0.55,1.85] → PL@1.90 → PD@1.95 → Death@2.00 |
| 0.4 | 0.100 | out_of_range | Inc[0.00,0.50] → PL[0.55,0.70] → Lock[0.75,1.80] → PL[1.85,1.95] → Death@2.00 |
| 0.4 | 0.125 | out_of_range | Inc[0.00,0.60] → PL[0.65,0.70] → Inc@0.75 → PL[0.80,0.90] → Lock[0.95,1.80] → PL[1.85,1.95] → Death@2.00 |
| 0.4 | 0.150 | out_of_range | Inc[0.00,0.80] → PL@0.85 → Inc[0.90,1.00] → PL[1.05,1.20] → Lock[1.25,1.65] → PL[1.70,1.95] → Death@2.00 |
| 0.4 | 0.175 | out_of_range | Inc[0.00,1.15] → PL[1.20,1.40] → Inc[1.45,1.50] → PL[1.55,1.60] → Inc@1.65 → PL[1.70,1.95] → Death@2.00 |

7 out-of-range columns already contain at least one near-zero oscillator before the scan ends. The predicted K was not reached, so these are not scored as formula errors. They do show that arrest is possible below an unscanned prediction.

| p | δ | K_c,MF | K_max | K_near | K_c−K_near | First non-Incoherence |
| --- | --- | --- | --- | --- | --- | --- |
| 0.4 | 0.000 | 2.500 | 2.00 | 1.95 | 0.550 | Locking |
| 0.4 | 0.025 | 2.438 | 2.00 | 1.95 | 0.488 | Partial Locking |
| 0.4 | 0.075 | 2.312 | 2.00 | 1.95 | 0.363 | Partial Locking |
| 0.4 | 0.100 | 2.250 | 2.00 | 2.00 | 0.250 | Partial Locking |
| 0.4 | 0.125 | 2.188 | 2.00 | 2.00 | 0.188 | Partial Locking |
| 0.4 | 0.150 | 2.125 | 2.00 | 2.00 | 0.125 | Partial Locking |
| 0.4 | 0.175 | 2.062 | 2.00 | 1.95 | 0.112 | Partial Locking |

## 6. Near-zero onset

On a clean column, K_onset is the smallest scanned K at which W3 has at least one oscillator with |ρ|<0.001. ΔK = K_onset − (1−δ)/p and Δ(pK) = p K_onset − (1−δ). The label compares K_onset with the grid, using the strict rule in Section 2. Only δ=1 has a natural frequency below 0.001. There ω_0=0, and at K=0 one finds ρ=ω, so that oscillator is near zero with no coupling. That column is identified in the table when its coupling-induced onset, the first K with a near-zero oscillator of |ω|≥0.001, differs from the raw onset.

| p | On the equality grid point | First grid point above the prediction | One grid step later | More than one grid step later | Earlier than the prediction | Clean n |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 5 | 5 | 0 | 0 | 7 | 17 |
| 0.4 | 5 | 6 | 0 | 0 | 21 | 32 |
| 0.8 | 3 | 22 | 0 | 0 | 2 | 27 |
| 1.0 | 1 | 30 | 0 | 0 | 0 | 31 |

| p | δ | K_c,MF | K_near | Grid label | Δ(pK) | n | Prefix run | Prefix set | K_coupling | State returns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 0.600 | 2.000 | 1.90 | Earlier than the prediction | -0.0200 | 1 | 0 | no | 1.90 | no |
| 0.2 | 0.625 | 1.875 | 1.80 | Earlier than the prediction | -0.0150 | 1 | 0 | no | 1.80 | no |
| 0.2 | 0.650 | 1.750 | 1.70 | Earlier than the prediction | -0.0100 | 2 | 0 | no | 1.70 | no |
| 0.2 | 0.675 | 1.625 | 1.55 | Earlier than the prediction | -0.0150 | 1 | 0 | no | 1.55 | no |
| 0.2 | 0.700 | 1.500 | 1.45 | Earlier than the prediction | -0.0100 | 1 | 0 | no | 1.45 | no |
| 0.2 | 0.725 | 1.375 | 1.35 | Earlier than the prediction | -0.0050 | 2 | 0 | no | 1.35 | no |
| 0.2 | 0.750 | 1.250 | 1.25 | On the equality grid point | 0.0000 | 2 | 0 | no | 1.25 | no |
| 0.2 | 0.775 | 1.125 | 1.10 | Earlier than the prediction | -0.0050 | 2 | 0 | no | 1.10 | no |
| 0.2 | 0.800 | 1.000 | 1.00 | On the equality grid point | 0.0000 | 2 | 0 | no | 1.00 | no |
| 0.2 | 0.825 | 0.875 | 0.90 | First grid point above the prediction | 0.0050 | 4 | 0 | no | 0.90 | no |
| 0.2 | 0.850 | 0.750 | 0.75 | On the equality grid point | 0.0000 | 1 | 0 | no | 0.75 | no |
| 0.2 | 0.875 | 0.625 | 0.65 | First grid point above the prediction | 0.0050 | 3 | 0 | no | 0.65 | no |
| 0.2 | 0.900 | 0.500 | 0.50 | On the equality grid point | 0.0000 | 1 | 0 | no | 0.50 | no |
| 0.2 | 0.925 | 0.375 | 0.40 | First grid point above the prediction | 0.0050 | 3 | 3 | yes | 0.40 | no |
| 0.2 | 0.950 | 0.250 | 0.30 | First grid point above the prediction | 0.0100 | 6 | 4 | no | 0.30 | no |
| 0.2 | 0.975 | 0.125 | 0.15 | First grid point above the prediction | 0.0050 | 3 | 3 | yes | 0.15 | no |
| 0.2 | 1.000 | 0.000 | 0.00 | On the equality grid point | 0.0000 | 1 | 1 | yes | 0.05 | no |
| 0.4 | 0.225 | 1.938 | 1.85 | Earlier than the prediction | -0.0350 | 2 | 1 | no | 1.85 | no |
| 0.4 | 0.250 | 1.875 | 1.80 | Earlier than the prediction | -0.0300 | 2 | 1 | no | 1.80 | no |
| 0.4 | 0.275 | 1.812 | 1.75 | Earlier than the prediction | -0.0250 | 3 | 1 | no | 1.75 | no |
| 0.4 | 0.300 | 1.750 | 1.70 | Earlier than the prediction | -0.0200 | 3 | 1 | no | 1.70 | no |
| 0.4 | 0.325 | 1.688 | 1.65 | Earlier than the prediction | -0.0150 | 3 | 1 | no | 1.65 | no |
| 0.4 | 0.350 | 1.625 | 1.60 | Earlier than the prediction | -0.0100 | 4 | 1 | no | 1.60 | no |
| 0.4 | 0.375 | 1.562 | 1.50 | Earlier than the prediction | -0.0250 | 2 | 1 | no | 1.50 | no |
| 0.4 | 0.400 | 1.500 | 1.45 | Earlier than the prediction | -0.0200 | 2 | 1 | no | 1.45 | no |
| 0.4 | 0.425 | 1.437 | 1.40 | Earlier than the prediction | -0.0150 | 2 | 1 | no | 1.40 | no |
| 0.4 | 0.450 | 1.375 | 1.35 | Earlier than the prediction | -0.0100 | 2 | 1 | no | 1.35 | no |
| 0.4 | 0.475 | 1.312 | 1.30 | Earlier than the prediction | -0.0050 | 2 | 1 | no | 1.30 | no |
| 0.4 | 0.500 | 1.250 | 1.20 | Earlier than the prediction | -0.0200 | 2 | 1 | no | 1.20 | no |
| 0.4 | 0.525 | 1.187 | 1.15 | Earlier than the prediction | -0.0150 | 2 | 1 | no | 1.15 | no |
| 0.4 | 0.550 | 1.125 | 1.10 | Earlier than the prediction | -0.0100 | 2 | 1 | no | 1.10 | no |
| 0.4 | 0.575 | 1.062 | 1.05 | Earlier than the prediction | -0.0050 | 2 | 1 | no | 1.05 | no |
| 0.4 | 0.600 | 1.000 | 1.00 | On the equality grid point | 0.0000 | 3 | 1 | no | 1.00 | no |
| 0.4 | 0.625 | 0.938 | 0.90 | Earlier than the prediction | -0.0150 | 2 | 1 | no | 0.90 | no |
| 0.4 | 0.650 | 0.875 | 0.85 | Earlier than the prediction | -0.0100 | 2 | 1 | no | 0.85 | no |
| 0.4 | 0.675 | 0.812 | 0.80 | Earlier than the prediction | -0.0050 | 2 | 1 | no | 0.80 | no |
| 0.4 | 0.700 | 0.750 | 0.75 | On the equality grid point | 0.0000 | 2 | 1 | no | 0.75 | no |
| 0.4 | 0.725 | 0.688 | 0.70 | First grid point above the prediction | 0.0050 | 4 | 1 | no | 0.70 | no |
| 0.4 | 0.750 | 0.625 | 0.60 | Earlier than the prediction | -0.0100 | 1 | 1 | yes | 0.60 | no |
| 0.4 | 0.775 | 0.562 | 0.55 | Earlier than the prediction | -0.0050 | 2 | 1 | no | 0.55 | no |
| 0.4 | 0.800 | 0.500 | 0.50 | On the equality grid point | 0.0000 | 2 | 1 | no | 0.50 | no |
| 0.4 | 0.825 | 0.438 | 0.45 | First grid point above the prediction | 0.0050 | 4 | 3 | no | 0.45 | no |
| 0.4 | 0.850 | 0.375 | 0.40 | First grid point above the prediction | 0.0100 | 6 | 6 | yes | 0.40 | no |
| 0.4 | 0.875 | 0.312 | 0.30 | Earlier than the prediction | -0.0050 | 1 | 1 | yes | 0.30 | no |
| 0.4 | 0.900 | 0.250 | 0.25 | On the equality grid point | 0.0000 | 2 | 1 | no | 0.25 | no |
| 0.4 | 0.925 | 0.187 | 0.20 | First grid point above the prediction | 0.0050 | 3 | 3 | yes | 0.20 | no |
| 0.4 | 0.950 | 0.125 | 0.15 | First grid point above the prediction | 0.0100 | 6 | 6 | yes | 0.15 | no |
| 0.4 | 0.975 | 0.063 | 0.10 | First grid point above the prediction | 0.0150 | 7 | 7 | yes | 0.10 | no |
| 0.4 | 1.000 | 0.000 | 0.00 | On the equality grid point | 0.0000 | 1 | 1 | yes | 0.05 | no |
| 0.8 | 0.325 | 0.844 | 0.85 | First grid point above the prediction | 0.0050 | 8 | 1 | no | 0.85 | no |
| 0.8 | 0.350 | 0.812 | 0.80 | Earlier than the prediction | -0.0100 | 1 | 0 | no | 0.80 | no |
| 0.8 | 0.375 | 0.781 | 0.80 | First grid point above the prediction | 0.0150 | 24 | 5 | no | 0.80 | no |
| 0.8 | 0.425 | 0.719 | 0.75 | First grid point above the prediction | 0.0250 | 37 | 9 | no | 0.75 | no |
| 0.8 | 0.450 | 0.688 | 0.70 | First grid point above the prediction | 0.0100 | 9 | 1 | no | 0.70 | no |
| 0.8 | 0.475 | 0.656 | 0.65 | Earlier than the prediction | -0.0050 | 1 | 0 | no | 0.65 | no |
| 0.8 | 0.500 | 0.625 | 0.65 | First grid point above the prediction | 0.0200 | 23 | 9 | no | 0.65 | no |
| 0.8 | 0.525 | 0.594 | 0.60 | First grid point above the prediction | 0.0050 | 2 | 1 | no | 0.60 | no |
| 0.8 | 0.550 | 0.562 | 0.60 | First grid point above the prediction | 0.0300 | 32 | 26 | no | 0.60 | no |
| 0.8 | 0.575 | 0.531 | 0.55 | First grid point above the prediction | 0.0150 | 11 | 5 | no | 0.55 | no |
| 0.8 | 0.600 | 0.500 | 0.50 | On the equality grid point | 0.0000 | 2 | 1 | no | 0.50 | no |
| 0.8 | 0.625 | 0.469 | 0.50 | First grid point above the prediction | 0.0250 | 23 | 22 | no | 0.50 | no |
| 0.8 | 0.650 | 0.437 | 0.45 | First grid point above the prediction | 0.0100 | 6 | 5 | no | 0.45 | no |
| 0.8 | 0.675 | 0.406 | 0.45 | First grid point above the prediction | 0.0350 | 29 | 26 | no | 0.45 | no |
| 0.8 | 0.700 | 0.375 | 0.40 | First grid point above the prediction | 0.0200 | 13 | 9 | no | 0.40 | no |
| 0.8 | 0.725 | 0.344 | 0.35 | First grid point above the prediction | 0.0050 | 2 | 1 | no | 0.35 | no |
| 0.8 | 0.750 | 0.312 | 0.35 | First grid point above the prediction | 0.0300 | 22 | 22 | yes | 0.35 | no |
| 0.8 | 0.775 | 0.281 | 0.30 | First grid point above the prediction | 0.0150 | 9 | 9 | yes | 0.30 | no |
| 0.8 | 0.800 | 0.250 | 0.25 | On the equality grid point | 0.0000 | 2 | 1 | no | 0.25 | no |
| 0.8 | 0.825 | 0.219 | 0.25 | First grid point above the prediction | 0.0250 | 16 | 13 | no | 0.25 | no |
| 0.8 | 0.850 | 0.188 | 0.20 | First grid point above the prediction | 0.0100 | 6 | 5 | no | 0.20 | no |
| 0.8 | 0.875 | 0.156 | 0.20 | First grid point above the prediction | 0.0350 | 22 | 22 | yes | 0.20 | no |
| 0.8 | 0.900 | 0.125 | 0.15 | First grid point above the prediction | 0.0200 | 11 | 9 | no | 0.15 | no |
| 0.8 | 0.925 | 0.094 | 0.10 | First grid point above the prediction | 0.0050 | 3 | 3 | yes | 0.10 | no |
| 0.8 | 0.950 | 0.063 | 0.10 | First grid point above the prediction | 0.0300 | 17 | 17 | yes | 0.10 | no |
| 0.8 | 0.975 | 0.031 | 0.05 | First grid point above the prediction | 0.0150 | 8 | 8 | yes | 0.05 | no |
| 0.8 | 1.000 | 0.000 | 0.00 | On the equality grid point | 0.0000 | 1 | 1 | yes | 0.05 | no |
| 1.0 | 0.250 | 0.750 | 0.80 | First grid point above the prediction | 0.0500 | 199 | 199 | yes | 0.80 | no |
| 1.0 | 0.275 | 0.725 | 0.75 | First grid point above the prediction | 0.0250 | 60 | 60 | yes | 0.75 | no |
| 1.0 | 0.300 | 0.700 | 0.75 | First grid point above the prediction | 0.0500 | 130 | 130 | yes | 0.75 | no |
| 1.0 | 0.325 | 0.675 | 0.70 | First grid point above the prediction | 0.0250 | 48 | 48 | yes | 0.70 | no |
| 1.0 | 0.350 | 0.650 | 0.70 | First grid point above the prediction | 0.0500 | 99 | 99 | yes | 0.70 | no |
| 1.0 | 0.375 | 0.625 | 0.65 | First grid point above the prediction | 0.0250 | 40 | 40 | yes | 0.65 | no |
| 1.0 | 0.400 | 0.600 | 0.65 | First grid point above the prediction | 0.0500 | 81 | 81 | yes | 0.65 | no |
| 1.0 | 0.425 | 0.575 | 0.60 | First grid point above the prediction | 0.0250 | 34 | 34 | yes | 0.60 | no |
| 1.0 | 0.450 | 0.550 | 0.60 | First grid point above the prediction | 0.0500 | 69 | 69 | yes | 0.60 | no |
| 1.0 | 0.475 | 0.525 | 0.55 | First grid point above the prediction | 0.0250 | 30 | 30 | yes | 0.55 | no |
| 1.0 | 0.500 | 0.500 | 0.55 | First grid point above the prediction | 0.0500 | 60 | 60 | yes | 0.55 | no |
| 1.0 | 0.525 | 0.475 | 0.50 | First grid point above the prediction | 0.0250 | 27 | 27 | yes | 0.50 | no |
| 1.0 | 0.550 | 0.450 | 0.50 | First grid point above the prediction | 0.0500 | 53 | 53 | yes | 0.50 | no |
| 1.0 | 0.575 | 0.425 | 0.45 | First grid point above the prediction | 0.0250 | 24 | 24 | yes | 0.45 | no |
| 1.0 | 0.600 | 0.400 | 0.45 | First grid point above the prediction | 0.0500 | 48 | 48 | yes | 0.45 | no |
| 1.0 | 0.625 | 0.375 | 0.40 | First grid point above the prediction | 0.0250 | 22 | 22 | yes | 0.40 | no |
| 1.0 | 0.650 | 0.350 | 0.40 | First grid point above the prediction | 0.0500 | 43 | 43 | yes | 0.40 | no |
| 1.0 | 0.675 | 0.325 | 0.35 | First grid point above the prediction | 0.0250 | 20 | 20 | yes | 0.35 | no |
| 1.0 | 0.700 | 0.300 | 0.35 | First grid point above the prediction | 0.0500 | 39 | 39 | yes | 0.35 | no |
| 1.0 | 0.725 | 0.275 | 0.30 | First grid point above the prediction | 0.0250 | 19 | 19 | yes | 0.30 | no |
| 1.0 | 0.750 | 0.250 | 0.30 | First grid point above the prediction | 0.0500 | 36 | 36 | yes | 0.30 | no |
| 1.0 | 0.775 | 0.225 | 0.25 | First grid point above the prediction | 0.0250 | 17 | 17 | yes | 0.25 | no |
| 1.0 | 0.800 | 0.200 | 0.25 | First grid point above the prediction | 0.0500 | 34 | 34 | yes | 0.25 | no |
| 1.0 | 0.825 | 0.175 | 0.20 | First grid point above the prediction | 0.0250 | 16 | 16 | yes | 0.20 | no |
| 1.0 | 0.850 | 0.150 | 0.20 | First grid point above the prediction | 0.0500 | 31 | 31 | yes | 0.20 | no |
| 1.0 | 0.875 | 0.125 | 0.15 | First grid point above the prediction | 0.0250 | 15 | 15 | yes | 0.15 | no |
| 1.0 | 0.900 | 0.100 | 0.15 | First grid point above the prediction | 0.0500 | 29 | 29 | yes | 0.15 | no |
| 1.0 | 0.925 | 0.075 | 0.10 | First grid point above the prediction | 0.0250 | 14 | 14 | yes | 0.10 | no |
| 1.0 | 0.950 | 0.050 | 0.10 | First grid point above the prediction | 0.0500 | 28 | 28 | yes | 0.10 | no |
| 1.0 | 0.975 | 0.025 | 0.05 | First grid point above the prediction | 0.0250 | 14 | 14 | yes | 0.05 | no |
| 1.0 | 1.000 | 0.000 | 0.00 | On the equality grid point | 0.0000 | 1 | 1 | yes | 0.05 | no |

## 7. Partial Death onset

The raw criterion and the final label are separate. On the clean columns the question is where a 15-oscillator run begins, not whether Eq. (5) itself uses the number 15. K_15(δ)=1−δ+28δ/999 is only the p=1 frequency of the fifteenth oscillator. It is not rederived for ER graphs.

| p | Clean | Raw PD, first K above | Raw PD, +1 step | Raw PD, >1 step | Raw PD early | Raw PD on equality | No PD label | Raw K ≠ label K | Raw PD at K_near |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 17 | 0 | 0 | 14 | 0 | 0 | 3 | 0 | 0 |
| 0.4 | 32 | 0 | 12 | 20 | 0 | 0 | 0 | 0 | 0 |
| 0.8 | 27 | 12 | 15 | 0 | 0 | 0 | 0 | 0 | 7 |
| 1.0 | 31 | 29 | 2 | 0 | 0 | 0 | 0 | 0 | 28 |

A raw run can be satisfied while the label is Death, when the near-zero fraction is already 1, or Locking, when the whole profile range is at most 0.01. Those overrides are the raw-versus-label disagreements. They are not removed from the raw count.

Eq. (5) is the first oscillator, not the fifteenth. At p=0.2 the first near-zero set has only a few oscillators, so a run of 15 is a later event. The 15-oscillator run lies more than one grid step after (1−δ)/p on 14 of the 17 clean columns. It is never reached inside the scan for δ=0.600, δ=0.625, δ=0.650. Those columns are not counted as failures of the first-oscillator comparison. δ=0.60 is also the upper edge of the p=0.2 scan, with K_c=K_max=2.

## 8. Scaled residual analysis

The same raw K step, 0.05, is a scaled step p×0.05: 0.010 at p=0.2, 0.020 at p=0.4, 0.040 at p=0.8, and 0.050 at p=1. A residual is therefore read in units of that local step. No fixed absolute cutoff is used as a pass or fail.

| p | Clean onsets | Median Δ(pK) | Min | Max | Median Δ(pK)/(p×0.05) | Median finite-N residual |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 17 | 0.0000 | -0.0200 | 0.0100 | 0.00 | -0.0002 |
| 0.4 | 32 | -0.0075 | -0.0350 | 0.0150 | -0.37 | -0.0079 |
| 0.8 | 27 | 0.0150 | -0.0100 | 0.0350 | 0.38 | 0.0146 |
| 1.0 | 31 | 0.0250 | 0.0000 | 0.0500 | 0.50 | 0.0249 |

The δ values that are clean at p=0.4, 0.8, and 1 simultaneously number 27. p=0.2 is included only where it is also clean. This is the overlap on which a p trend can be looked at without mixing different δ windows. One network and one initial condition still make it a single-trial comparison.

| p | Overlap n | Median Δ(pK) | First K above | +1 step | Early | >1 step | Equality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 17 | 0.0000 | 5 | 0 | 7 | 0 | 5 |
| 0.4 | 27 | -0.0050 | 6 | 0 | 16 | 0 | 5 |
| 0.8 | 27 | 0.0150 | 22 | 0 | 2 | 0 | 3 |
| 1.0 | 27 | 0.0250 | 26 | 0 | 0 | 0 | 1 |

An early onset is not all one thing. `Adjacent` means the onset is the greatest scanned K that is still below (1−δ)/p, so the signed distance is smaller than one grid step. `Further` means the onset is at least one additional step before that neighbor.

| p | Early clean columns | Adjacent lower grid point | At least one step earlier than that | Scaled step p×0.05 |
| --- | --- | --- | --- | --- |
| 0.2 | 7 | 4 | 3 | 0.010 |
| 0.4 | 21 | 17 | 4 | 0.020 |
| 0.8 | 2 | 2 | 0 | 0.040 |
| 1.0 | 0 | 0 | 0 | 0.050 |

The finite-N column shifts every residual by about p K_onset/N. On this grid that shift does not move an onset from one K bin to another. Agreement with pK=1−δ and agreement with the mean-degree correction are the same statement at the available resolution.

On the shared clean widths the median Δ(pK) is 0.0000 at p=0.2, -0.0050 at p=0.4, 0.0150 at p=0.8, 0.0250 at p=1.0. Those values are not a steady growth of the residual as p decreases. The p=0.2 median can sit near zero because early and late columns cancel, and its clean widths are only the large-δ end of the scan.

## 9. Network degree exploration

The graphs above are the graphs named by the saved seeds and hashes. Index 0 is the slowest oscillator at every δ, and the graph does not change with K or δ. A degree quoted for “the first arrested oscillator” is therefore the degree of a fixed node whenever the near-zero set is always that same node. Repeating it across δ does not create an ensemble.

| p | Clean onsets | Prefix of ω | Includes slowest | Higher-ω, higher-degree inversion | Median overlap with lowest ω | Median overlap with lowest ω/d | Median norm. degree of the set |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 17 | 3/17 | 4/17 | 14/17 | 0.000 | 1.000 | 1.064 |
| 0.4 | 32 | 7/32 | 32/32 | 25/32 | 0.500 | 1.000 | 1.049 |
| 0.8 | 27 | 7/27 | 25/27 | 20/27 | 0.909 | 1.000 | 1.001 |
| 1.0 | 31 | 31/31 | 31/31 | 0/31 | 1.000 | 1.000 | 1.000 |

Overlap is the fraction of the first near-zero set that also lies among the same number of oscillators with the smallest ω, or the smallest ω/d. ω/d is an exploratory score, not an arrest threshold. An inversion is a still-rotating oscillator at lower ω than some near-zero oscillator, with the near-zero oscillator at higher degree. Normalized degree uses (N−1)p, the expected degree, not the realized mean.

On the saved p=0.2 graph the slowest oscillator, index 0, has degree 191 (normalized 0.956). It is outside the first near-zero set in 13 of 17 clean columns. The nodes that do appear are the same ones across δ: rank 8 in 12/17 columns (degree 217), rank 2 in 12/17 columns (degree 208), rank 1 in 5/17 columns (degree 198), rank 0 in 4/17 columns (degree 191). Index 8 has degree 217, against 191 for the slowest oscillator, while its natural frequency stays close to ω_min. The median overlap of the first set with the lowest ω is 0. The median overlap with the lowest ω/d is 1. On this single graph the exploratory score matches the first set and the pure frequency order does not. That is not an ensemble result and it is not a derived arrest law. At p=0.4 the slowest oscillator is inside 32 of 32 clean first sets, but the set is usually not a frequency prefix: index 1 has degree 392, below index 0 (418) and index 2 (420). Index 51, degree 430, joins that set on 4 widths. At p=1 every degree is 999 and every clean first set is the low-frequency prefix, so degree has nothing to reorder.

## 10. Finite-time sensitivity

A W3 near-zero value is a rotation number over a 2000-time-unit window. It is not an infinite-time rest point. For each clean column the first K with a positive near-zero count is computed separately on W1, W2, and W3. At the W3 onset cell, each near-zero oscillator is then marked stable when |ρ|<0.001 in all three windows, W3-only when it misses both earlier windows, and mixed otherwise. Drift means the three rotation numbers span more than 0.01.

| p | Clean | Same K on W1–W3 | W3 later | W3 earlier | Only W3 has an onset | Oscillators stable | W3 only | Mixed | ρ span > 0.01 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 17 | 17 | 0 | 0 | 0 | 36/36 | 0/36 | 0/36 | 0/36 |
| 0.4 | 32 | 32 | 0 | 0 | 0 | 85/85 | 0/85 | 0/85 | 0/85 |
| 0.8 | 27 | 27 | 0 | 0 | 0 | 339/340 | 1/340 | 0/340 | 0/340 |
| 1.0 | 31 | 31 | 0 | 0 | 0 | 1380/1380 | 0/1380 | 0/1380 | 0/1380 |

Column counts use the first K at which the near-zero count becomes positive. Oscillator counts pool every oscillator in those onset cells. The same node can enter more than one δ, so the oscillator totals are not independent samples.

## 11. Limitations

There is one trial, one initial-phase draw, and one nested family of ER graphs. A residual at a given p is a property of that draw. It does not estimate the distribution over networks or initial data. The K step is 0.05. Onsets are grid events, not interpolated critical values. Windows end at t=14000. A small rotation number can still drift, and a true rest point can be masked by a slow transient. The classifier cuts were not adjusted to Eq. (5). The annealed factor pK ignores both the (N−1)/N normalization and quenched degree variation. The finite-N expression restores only the expected mean degree. Out-of-range predictions, especially at p=0.2 and small δ, are censored. Absence of Partial Death in a censored column is not evidence against Eq. (5). Competing and interrupted columns leave the incoherent branch before the arrest Eq. (5) describes, so they are not pooled into the residual. δ=0 has no frequency order. The saved files do not contain a measured mean-field amplitude σ, so a change of the mean field after the first oscillators stop is not measured here.

## 12. Conclusion

The p=1 baseline reproduces the previous Eq. (5) note. On the new |ρ|<0.001 definition, the 30 clean widths δ=0.25 through 0.975 all place the first near-zero oscillator on the first scanned K strictly above 1−δ. The single equality point in the p=1 clean count is δ=1, where ω_min=0 and the K=0 rotation number is the free frequency. No clean p=1 column is early, and none is more than one grid step late. The positive Δ(pK) values 0.025 and 0.050 are the distance from 1−δ up to the next grid point, which is the resolution of the scan.

At p=0.8, 22 clean columns are on that same first strict grid point, 3 sit on the equality point, and 2 are early. Of the early ones, 2 are the adjacent lower grid point and 0 are earlier than that. None is an extra step late. For the clean large-δ branch, pK_onset tracks 1−δ to within one coupling step.

At p=0.4 the same count is 21 early, 6 on the first point above, and 5 on the equality point, out of 32 clean columns. 17 of the early columns are only the neighboring K below the prediction. 4 are at least one further step early; the most negative Δ(pK) in the table is about −0.035, just under two scaled steps of 0.020. The annealed line is still the right neighborhood, and the bias on this draw is early rather than late. It is not a multi-step collapse of the scaling, and it is not a single-trial proof that the formula is shifted by a definite amount.

At p=0.2 only δ≥0.60 has (1−δ)/p inside the scan. Those 17 clean columns split into 5 on the first point above, 5 on the equality point, and 7 early (4 adjacent, 3 one further step). There is no extra-step delay. The K location is within about two grid steps of (1−δ)/p. The identity of the arrested oscillator is a separate and clearer departure from the all-to-all picture: the slowest node is outside the first near-zero set in most of these columns, and the nodes that stop are the high-degree nodes already named in Section 9. δ≤0.575 at p=0.2 is out of range. No near-zero onset there would not have been a failed test.

26 columns are competing. Eq. (5) is not scored on them. They are the small-δ locking sequences at p=1 (δ≤0.225) and p=0.8 (δ≤0.30 and δ=0.40), plus p=0.4 at δ=0.05 and δ=0.20. Several of those sequences are non-monotonic; the returned Incoherence cells were kept. 31 columns are out of range, including every p=0.2 width with δ≤0.575 and seven small-δ widths at p=0.4. Seven of the out-of-range columns do contain a near-zero oscillator before the scan ends. Because the predicted K was never reached, that observation is reported and not entered as a residual.

Network heterogeneity is visible on this one nested family. At p=0.2 the slowest oscillator is missing from 13 of 17 clean first sets, while the recurring arrested nodes have higher degree than it does. The lowest-ω/d set matches those arrests; the lowest-ω set does not. At p=1 the same comparison is empty, because every degree is identical and every clean set is a low-frequency prefix. One graph cannot say how often a different ER draw would move the onset.

Finite time does not explain the early columns. On every clean column the first positive near-zero count occurs at the same K in W1, W2, and W3, and essentially every oscillator in that onset cell is below 0.001 in all three windows. The (N−1)/N factor and the expected-degree correction are both far smaller than a 0.05 step, so they do not account for an onset that is one or two steps early. What remains is the annealed replacement A_ij≈p, the quenched degrees of this draw, and the finite-time meaning of |ρ|<0.001. The windows rule out a late-only transient. They do not separate the annealed error from this graph's degree sequence. The saved scan therefore supports pK_onset≈1−δ to within one coupling step on the clean p=1 and p=0.8 branches, supports it only with an early bias of up to about two steps on the clean p=0.4 and large-δ p=0.2 branches, and does not test it on the locking side or outside the scanned K range. It does not turn the annealed formula into an exact ER threshold.

Figure: `eq5_er_scaling_validation.png`. Script: `eq5_er_scaling/eq5_er_scaling_validation.py`. No ODE was rerun, no Stage 1 file was edited, and no classifier threshold was changed.
