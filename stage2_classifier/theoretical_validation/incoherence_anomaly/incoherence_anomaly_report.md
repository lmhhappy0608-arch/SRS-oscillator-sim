# Incoherence anomaly diagnostic

Operational Incoherence on the saved p=1 grid is compared with the printed Eq. (7) Hopf truncation. The linear instability of the incoherent state, the finite-time rotation-number profile, and the Classifier v1 label are kept separate. A green cell above the truncation is not treated as a misclassification by default.

## 1. Data and classifier provenance

The Stage 1 index has 6964 rows and 6962 completed runs. The completed p=1 slice has 1741 profiles, all with N=1000 and trial 0. One provenance group covers that slice: rk4, dt=0.1, t from 0 to 14000, master_seed=0, seed_recipe=sha256-v1, paired randomness, network_seed=8250318217634423533, initial_phase_seed=1364563909611740937, frequency rule deterministic_linspace_1_minus_delta_to_1_plus_delta. Network realization 0 and initial condition 0 are shared. The p=1 adjacency has diagonal sum 0. Regenerating it from the stored network seed matches the archived hash. W1=[8000, 10000], W2=[10000, 12000], and W3=[12000, 14000]. W3 assigns the primary state. The archived classification_results.csv contains the original 732-run pilot and does not cover this full p=1 rectangle. Labels used here were rebuilt in memory with Classifier v1.0.0-pilot and were not written back.

Classifier v1.0.0-pilot reads alpha=2, beta=1, and Method C minimum segment length 4 from the archived config. Near-zero means |ρ|<0.001. Death requires near-zero fraction 1. Locking requires the full-profile ρ range ≤ 0.01. Partial Death requires at least 15 consecutive near-zero oscillators after those two rules. Partial Locking requires at least one Method C constant segment that passes Rule 3 (at least 15 oscillators, relative ω span ≥ 0.02, ρ span ≤ 0.02) with |mean ρ|≥0.001. Incoherence is the remaining label. The priority is Death, Locking, Partial Death, Partial Locking, Incoherence. Cells with K>1 were screened from the index and not loaded: for every δ in (0, 0.20], both 1−δ and K_exist are below 1, so K≥1 cannot satisfy the strict inequalities. Profiles with K≤1 were loaded (861 cells). Method C ran on the 330 W3 profiles that were not already Death, Locking, or Partial Death. Diagnosed cells were then segmented on W1, W2, and W3.

State counts on the loaded K≤1 slice: {'Locking': 88, 'Death': 103, 'Incoherence': 329, 'Partial Locking': 1, 'Partial Death': 340}.

## 2. Theoretical candidate definition

Eq. (7) is the printed product truncation, reused from the Eq. (7) validation and not refit:

K_H^(7)(δ) = (8δ/π) [ 1 + 16δ²/π² + 16(π²+80)δ⁴/π⁴ ].

Eq. (5) on the incoherent branch is K_c = 1−δ. K_exist(δ) is the death-equilibrium existence value from the same function used in the death-boundary validation. At δ=0.20 that function returns the previously checked value 0.7902 within 5×10⁻⁴. Eq. (6) roots are the values already printed for δ=0.05, 0.10, 0.15, and 0.20. Widths that were not solved, or that had no valid root, do not receive a K−K_H^(6) entry. No boundary was interpolated or refit.

| δ | K_H^(7) | K_H^(6) | K_H^(6) − K_H^(7) | 1 − δ | K_exist |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.025 | 0.06373 | not solved | NA | 0.975 | 0.7701 |
| 0.050 | 0.12785 | 0.12785 | -0.00000 | 0.950 | 0.7711 |
| 0.075 | 0.19282 | not solved | NA | 0.925 | 0.7727 |
| 0.100 | 0.25915 | 0.25920 | +0.00005 | 0.900 | 0.7749 |
| 0.125 | 0.32752 | not solved | NA | 0.875 | 0.7778 |
| 0.150 | 0.39876 | 0.39991 | +0.00115 | 0.850 | 0.7813 |
| 0.175 | 0.47393 | not solved | NA | 0.825 | 0.7855 |
| 0.200 | 0.55435 | 0.56905 | +0.01470 | 0.800 | 0.7902 |

A strict candidate meets all five conditions: W3 state Incoherence, 0<δ≤0.20, K>K_H^(7)(δ), K<1−δ, and K<K_exist(δ). Equality on Eq. (7) is not a candidate. A strict candidate is marked near-boundary when a reliable Eq. (6) root exists and the excess K−K_H^(7) does not exceed that truncation gap by more than 10⁻⁴, or when K is within 0.001 of the Eq. (6) root. If no Eq. (6) root is available, the near-boundary mark requires K−K_H^(7)≤0.001. The mark keeps the truncation visible. It does not remove the cell from the strict count and it does not create a new state.

Named controls are δ=0.15, K=0.40; δ=0.20, K=0.60; δ=0.20, K=0.65; δ=0.20, K=0.70; and δ=0.20, K=0.75. They are diagnosed whether or not they meet the five conditions. A control enters the strict count only when the five conditions are true. The control list is not an inclusion rule.

## 3. Strict candidate counts and locations

The K≤1 slice has 56 Incoherence cells with 0<δ≤0.20. Of those, 4 have K>K_H^(7), 4 also have K<1−δ, and 4 also have K<K_exist. That last number is the strict-candidate count: 4. They occupy 2 delta value(s): δ=0.150, K from 0.40 to 0.40 (1 cell); δ=0.200, K from 0.60 to 0.75 (3 cells). Distances K−K_H^(7): minimum +0.00124, median +0.09565, maximum +0.19565. 3 strict candidates lie outside the near-boundary mark, and 1 is marked near-boundary.

| δ | K | K − K_H^(7) | K − K_H^(6) | K_exist | near-boundary | tags |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0.150 | 0.40 | +0.00124 | +0.00009 | 0.7813 | yes | Theoretically ambiguous near-boundary case; Possible missed local plateau; Temporally evolving profile; Broad drifting profile |
| 0.200 | 0.60 | +0.04565 | +0.03095 | 0.7902 | no | Near-global-locking |
| 0.200 | 0.70 | +0.14565 | +0.13095 | 0.7902 | no | Near-global-locking |
| 0.200 | 0.75 | +0.19565 | +0.18095 | 0.7902 | no | Flat majority with high-frequency tail |

delta=0.150 has the single K=0.40; delta=0.200 has K=0.60, 0.70, 0.75, with a gap larger than the 0.05 grid. 1 delta value(s) contribute only one candidate. 0 pair(s) of candidate deltas are adjacent on the 0.025 delta grid.

Named controls, kept out of the count above unless the five conditions already include them:

| δ | K | W3 state | strict | near-boundary | assigned control type |
| ---: | ---: | --- | --- | --- | --- |
| 0.15 | 0.40 | Incoherence | yes | yes | near-boundary control |
| 0.20 | 0.60 | Incoherence | yes | no | pre-locking control |
| 0.20 | 0.65 | Locking | no | no | global-locking control |
| 0.20 | 0.70 | Incoherence | yes | no | non-monotonic transition control |
| 0.20 | 0.75 | Incoherence | yes | no | non-monotonic transition control |

## 4. Rotation-number structural diagnostics

The 5–95% width and the central-90% range describe the bulk. An oscillator is counted as outside the main group when |ρ − median(ρ)| > 0.01, the archived locking range. A flat majority, used only as a descriptive tag, means at least 90% of oscillators lie inside that same band. Neither comparison replaces Rule 7, which still uses the full-profile range.

| δ | K | state | range | mean | median | near-zero | corr(ρ, ω) | 5–95% width | central 90% range | outside median±0.01 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.15 | 0.40 | Incoherence | 0.325488 | 0.9157 | 0.9030 | 0.0000 | 0.999 | 0.292700 | 0.292868 | 871 |
| 0.20 | 0.60 | Incoherence | 0.093474 | 0.7277 | 0.7256 | 0.0000 | 0.296 | 7.574e-05 | 7.211e-05 | 34 |
| 0.20 | 0.65 | Locking | 7.510e-05 | 0.6472 | 0.6472 | 0.0000 | -0.875 | 5.093e-05 | 5.092e-05 | 0 |
| 0.20 | 0.70 | Incoherence | 0.015726 | 0.5483 | 0.5485 | 0.0000 | -0.507 | 0.001088 | 0.001093 | 1 |
| 0.20 | 0.75 | Incoherence | 0.193888 | 0.4250 | 0.4155 | 0.0000 | 0.416 | 0.106953 | 0.106914 | 73 |

Among the 4 strict candidates, 2 have a bulk range at or below 0.01 while the full profile exceeds 0.01. 1 carries the broad-drifting tag. 1 carries the near-boundary tag.

## 5. Method C diagnostics

Constant segments are the archived candidates. Linear segments are not searched again and are not merged. A linear segment is described as an approximate plateau only when it meets the archived Rule 3 geometry: length at least 15, relative ω span at least 0.02, and ρ span at most 0.02, with |mean ρ| at least 0.001. A short piece of a slope is not counted. These comparisons do not change the classifier.

| δ | K | constant segments | validated nonzero | largest constant candidate | short range carrier |
| ---: | ---: | ---: | ---: | --- | --- |
| 0.15 | 0.40 | 13 | 0 | n=9, ω span=0.0024, ρ span=0.0016, mean=0.8677; length 9 < 15; relative omega span 0.0080 < 0.02 | linear n=115, ρ span=0.0365 |
| 0.20 | 0.60 | 0 | 0 | none | linear n=4, ρ span=0.0050 |
| 0.20 | 0.65 | 0 | 0 | none | linear n=6, ρ span=0.0000 |
| 0.20 | 0.70 | 0 | 0 | none | linear n=4, ρ span=0.0157 |
| 0.20 | 0.75 | 0 | 0 | none | linear n=4, ρ span=0.0053 |

- δ=0.15, K=0.40: 6 linear segment(s) meet the Rule 3 length, relative-span, and rho-span tests but are not constant candidates; the longest has length 77, relative omega span 0.076, rho span 0.0012, mean 0.9018, omega [0.9764, 0.9992]; 13 constant candidate(s) are shorter than 15 oscillators Tags: Theoretically ambiguous near-boundary case; Possible missed local plateau; Temporally evolving profile; Broad drifting profile.
- δ=0.20, K=0.60: 16 linear segment(s) meet the Rule 3 length, relative-span, and rho-span tests but are not constant candidates; the longest has length 133, relative omega span 0.132, rho span 0.0000, mean 0.7256, omega [0.8981, 0.9510]; sigma_hat=1.427e-10, so these linear pieces partition a flat majority; that is the noise-floor behavior already seen on Rule 7 Locking profiles, not a separate missed plateau Tags: Near-global-locking.
- δ=0.20, K=0.65: 17 linear segment(s) meet the Rule 3 length, relative-span, and rho-span tests but are not constant candidates; the longest has length 114, relative omega span 0.113, rho span 0.0000, mean 0.6472, omega [1.1055, 1.1508]; sigma_hat=7.304e-11, so these linear pieces partition a flat majority; that is the noise-floor behavior already seen on Rule 7 Locking profiles, not a separate missed plateau Tags: Global locking control.
- δ=0.20, K=0.70: 2 linear segment(s) meet the Rule 3 length, relative-span, and rho-span tests but are not constant candidates; the longest has length 24, relative omega span 0.023, rho span 0.0000, mean 0.5483, omega [1.0334, 1.0426]; the oscillator farthest from the median sits in a linear segment of length 4 whose rho span is 0.0157; sigma_hat=9.944e-10, so these linear pieces partition a flat majority; that is the noise-floor behavior already seen on Rule 7 Locking profiles, not a separate missed plateau Tags: Near-global-locking.
- δ=0.20, K=0.75: 3 linear segment(s) meet the Rule 3 length, relative-span, and rho-span tests but are not constant candidates; the longest has length 45, relative omega span 0.044, rho span 0.0000, mean 0.4151, omega [1.0979, 1.1155]; sigma_hat=3.364e-09, so these linear pieces partition a flat majority; that is the noise-floor behavior already seen on Rule 7 Locking profiles, not a separate missed plateau Tags: Flat majority with high-frequency tail.

Validated nonzero plateaus on the diagnosed Incoherence cells: 0. Strict candidates tagged as a possible missed local plateau: 1. A short segment that carries the global range is a failed member-count check, not a missed long plateau.

## 6. W1/W2/W3 temporal consistency

D is |ρ(W2)−ρ(W1)| and F is |ρ(W3)−ρ(W2)|, matched on the frequency-sorted oscillators. The review flag is the archived one: it is set when the three states differ or the validated nonzero-plateau counts differ. A small D and F, or a clear flag, does not prove convergence. The saved files also contain an order-parameter time series; it was not used here.

| δ | K | med D | max D | med F | max F | W1, W2, W3 states | ranges | plateau counts | coverages | review flag |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 0.15 | 0.40 | 0.001018 | 0.009272 | 0.003119 | 0.017806 | Partial Locking, Incoherence, Incoherence | 0.321441, 0.322599, 0.325488 | 1, 0, 0 | 0.022, 0.000, 0.000 | temporal_state_disagreement;temporal_plateau_count_disagreement |
| 0.20 | 0.60 | 1.605e-05 | 0.002838 | 9.592e-06 | 0.002954 | Incoherence, Incoherence, Incoherence | 0.094232, 0.094142, 0.093474 | 0, 0, 0 | 0.000, 0.000, 0.000 | clear |
| 0.20 | 0.65 | 5.920e-06 | 4.145e-05 | 3.946e-06 | 3.096e-05 | Locking, Locking, Locking | 1.521e-04, 1.085e-04, 7.510e-05 | 0, 0, 0 | 0.000, 0.000, 0.000 | clear |
| 0.20 | 0.70 | 7.833e-04 | 0.004496 | 9.201e-04 | 0.004335 | Incoherence, Incoherence, Incoherence | 0.016045, 0.020118, 0.015726 | 0, 0, 0 | 0.000, 0.000, 0.000 | clear |
| 0.20 | 0.75 | 0.001596 | 0.006022 | 0.001673 | 0.009174 | Incoherence, Incoherence, Incoherence | 0.194227, 0.194633, 0.193888 | 0, 0, 0 | 0.000, 0.000, 0.000 | clear |

3 of 4 strict candidates have the same archived state and the same validated-plateau count in all three windows. 1 strict candidate carries the temporally evolving tag.

## 7. Representative case studies

### Case A: δ=0.15, K=0.40

Control type on the named list, where applicable: near-boundary control. Strict candidate: yes. Near-boundary flag: yes. K_H^(7)=0.39876, K−K_H^(7)=+0.00124, K_H^(6) difference=+0.00009, truncation K_H^(6)−K_H^(7)=+0.00115, 1−δ=0.850, K_exist=0.7813.

W3 ρ lies in [0.7513, 1.0768], range 0.325488, mean 0.9157, median 0.9030, near-zero fraction 0.0000. Pearson correlation of ρ with ω is 0.999 and the least-squares slope is 1.0718 per unit frequency. The 5–95% ρ width is 0.292700. The central 90% of oscillators in frequency order have ρ range 0.292868. 129 oscillators lie within the archived locking range 0.01 of the median, and 871 lie outside it (50 in the low-frequency 5%, 771 in the interior, 50 in the high-frequency 5%). The lowest frequency among those outside points is ω=0.8500. The minimum is at ω=0.8500 and the maximum at ω=1.1500.

The W3 label is Incoherence. Near-zero fraction 0.0000 (Death requires 1), ρ range 0.325488 (Locking requires range ≤ 0.01), longest consecutive near-zero run 0 (Partial Death requires 15), and 0 validated nonzero plateau(s) (Partial Locking requires one). Sigma_hat is 2.591e-04.

Method C: 45 segments, 13 constant and 32 linear, 0 validated nonzero plateaus, plateau coverage 0.000. Largest constant candidate: n=9, ω span=0.0024, ρ span=0.0016, mean=0.8677; length 9 < 15; relative omega span 0.0080 < 0.02. Reading of the segment record: 6 linear segment(s) meet the Rule 3 length, relative-span, and rho-span tests but are not constant candidates; the longest has length 77, relative omega span 0.076, rho span 0.0012, mean 0.9018, omega [0.9764, 0.9992]; 13 constant candidate(s) are shorter than 15 oscillators. Validated nonzero plateaus by window — W1: n=22, relative ω span=0.021, ρ span=0.0025, mean=0.8989, ω=[0.9677, 0.9740].

Window changes: median(D)=0.001018, max(D)=0.009272, median(F)=0.003119, max(F)=0.017806. States W1/W2/W3 = Partial Locking / Incoherence / Incoherence. Ranges = 0.321441, 0.322599, 0.325488. Validated plateau counts = 1, 0, 0. Plateau coverages = 0.022, 0.000, 0.000. Archived temporal review flag: set (temporal_state_disagreement;temporal_plateau_count_disagreement). Agreement of the three window labels and a small pointwise change do not prove that the trajectory has converged.

Diagnostic tags, which are not new states: Theoretically ambiguous near-boundary case; Possible missed local plateau; Temporally evolving profile; Broad drifting profile.

W3 column for this δ, restricted to K≤1: Incoherence [0.00, 0.40] | Locking [0.45, 0.75] | Death [0.80, 1.00].

The excess over Eq. (7) is inside the Eq. (6) truncation, and the cell sits on the printed Eq. (6) root (K−K_H^(6)=+0.00009). The profile tracks frequency: correlation 0.999, slope 1.072, range 0.325488. W3 also has a linear segment that meets the Rule 3 length, relative-span, and ρ-span tests; the longest has n=77, relative ω span 0.076, ρ span 0.0012, mean 0.9018. Method C keeps it linear, so it is not a validated plateau. W1 validates 1 nonzero plateau, the first with n=22, relative ω span 0.021, mean ρ 0.8989, which is why the archived review flag is set. The shelf is visible on this trajectory and is close to the Rule 3 boundary. The cell remains a near-boundary strict candidate. It is not identified as a misclassification.

### Case B: δ=0.20, K=0.65

Control type on the named list, where applicable: global-locking control. Strict candidate: no. Near-boundary flag: no. K_H^(7)=0.55435, K−K_H^(7)=+0.09565, K_H^(6) difference=+0.08095, truncation K_H^(6)−K_H^(7)=+0.01470, 1−δ=0.800, K_exist=0.7902.

W3 ρ lies in [0.6472, 0.6473], range 7.510e-05, mean 0.6472, median 0.6472, near-zero fraction 0.0000. Pearson correlation of ρ with ω is -0.875 and the least-squares slope is -0.0001 per unit frequency. The 5–95% ρ width is 5.093e-05. The central 90% of oscillators in frequency order have ρ range 5.092e-05. 1000 oscillators lie within the archived locking range 0.01 of the median, and 0 lie outside it (0 in the low-frequency 5%, 0 in the interior, 0 in the high-frequency 5%). The minimum is at ω=1.2000 and the maximum at ω=0.8000.

The W3 label is Locking. Near-zero fraction 0.0000 (Death requires 1), ρ range 7.510e-05 (Locking requires range ≤ 0.01), longest consecutive near-zero run 0 (Partial Death requires 15), and 0 validated nonzero plateau(s) (Partial Locking requires one). Sigma_hat is 7.304e-11.

Method C: 43 segments, 0 constant and 43 linear, 0 validated nonzero plateaus, plateau coverage 0.000. Largest constant candidate: none. Reading of the segment record: 17 linear segment(s) meet the Rule 3 length, relative-span, and rho-span tests but are not constant candidates; the longest has length 114, relative omega span 0.113, rho span 0.0000, mean 0.6472, omega [1.1055, 1.1508]; sigma_hat=7.304e-11, so these linear pieces partition a flat majority; that is the noise-floor behavior already seen on Rule 7 Locking profiles, not a separate missed plateau. No window contains a validated nonzero plateau.

Window changes: median(D)=5.920e-06, max(D)=4.145e-05, median(F)=3.946e-06, max(F)=3.096e-05. States W1/W2/W3 = Locking / Locking / Locking. Ranges = 1.521e-04, 1.085e-04, 7.510e-05. Validated plateau counts = 0, 0, 0. Plateau coverages = 0.000, 0.000, 0.000. Archived temporal review flag: clear. Agreement of the three window labels and a small pointwise change do not prove that the trajectory has converged.

Diagnostic tags, which are not new states: Global locking control.

W3 column for this δ, restricted to K≤1: Incoherence [0.00, 0.60] | Locking [0.65, 0.65] | Incoherence [0.70, 0.75] | Death [0.80, 1.00].

This is the global-locking control on the non-monotonic column. The full-profile range 7.510e-05 is far below 0.01, so Rule 7 assigns Locking. Method C is not the source of that label. The same pattern was already reported for other Locking cells: a nearly flat nonzero profile can be fit entirely by linear segments when the noise estimate is tiny.

### Case C: δ=0.20, K=0.70

Control type on the named list, where applicable: non-monotonic transition control. Strict candidate: yes. Near-boundary flag: no. K_H^(7)=0.55435, K−K_H^(7)=+0.14565, K_H^(6) difference=+0.13095, truncation K_H^(6)−K_H^(7)=+0.01470, 1−δ=0.800, K_exist=0.7902.

W3 ρ lies in [0.5473, 0.5630], range 0.015726, mean 0.5483, median 0.5485, near-zero fraction 0.0000. Pearson correlation of ρ with ω is -0.507 and the least-squares slope is -0.0026 per unit frequency. The 5–95% ρ width is 0.001088. The central 90% of oscillators in frequency order have ρ range 0.001093. 999 oscillators lie within the archived locking range 0.01 of the median, and 1 lie outside it (0 in the low-frequency 5%, 0 in the interior, 1 in the high-frequency 5%). The lowest frequency among those outside points is ω=1.2000. The minimum is at ω=1.1996 and the maximum at ω=1.2000.

The W3 label is Incoherence. Near-zero fraction 0.0000 (Death requires 1), ρ range 0.015726 (Locking requires range ≤ 0.01), longest consecutive near-zero run 0 (Partial Death requires 15), and 0 validated nonzero plateau(s) (Partial Locking requires one). Sigma_hat is 9.944e-10.

Method C: 132 segments, 0 constant and 132 linear, 0 validated nonzero plateaus, plateau coverage 0.000. Largest constant candidate: none. Reading of the segment record: 2 linear segment(s) meet the Rule 3 length, relative-span, and rho-span tests but are not constant candidates; the longest has length 24, relative omega span 0.023, rho span 0.0000, mean 0.5483, omega [1.0334, 1.0426]; the oscillator farthest from the median sits in a linear segment of length 4 whose rho span is 0.0157; sigma_hat=9.944e-10, so these linear pieces partition a flat majority; that is the noise-floor behavior already seen on Rule 7 Locking profiles, not a separate missed plateau. No window contains a validated nonzero plateau.

Window changes: median(D)=7.833e-04, max(D)=0.004496, median(F)=9.201e-04, max(F)=0.004335. States W1/W2/W3 = Incoherence / Incoherence / Incoherence. Ranges = 0.016045, 0.020118, 0.015726. Validated plateau counts = 0, 0, 0. Plateau coverages = 0.000, 0.000, 0.000. Archived temporal review flag: clear. Agreement of the three window labels and a small pointwise change do not prove that the trajectory has converged.

Diagnostic tags, which are not new states: Near-global-locking.

W3 column for this δ, restricted to K≤1: Incoherence [0.00, 0.60] | Locking [0.65, 0.65] | Incoherence [0.70, 0.75] | Death [0.80, 1.00].

999 of 1000 oscillators are within 0.01 of the median, and the central-90% range is 0.001093. The full range is 0.015726, which is what keeps Rule 7 from firing. The segment record attributes that range to a short piece rather than to a long plateau that failed a span test. Method C did not validate a nonzero plateau, and the short piece is below the archived membership minimum, so this is not evidence that a long plateau was overlooked.

### Case D: δ=0.20, K=0.75

Control type on the named list, where applicable: non-monotonic transition control. Strict candidate: yes. Near-boundary flag: no. K_H^(7)=0.55435, K−K_H^(7)=+0.19565, K_H^(6) difference=+0.18095, truncation K_H^(6)−K_H^(7)=+0.01470, 1−δ=0.800, K_exist=0.7902.

W3 ρ lies in [0.4148, 0.6087], range 0.193888, mean 0.4250, median 0.4155, near-zero fraction 0.0000. Pearson correlation of ρ with ω is 0.416 and the least-squares slope is 0.1274 per unit frequency. The 5–95% ρ width is 0.106953. The central 90% of oscillators in frequency order have ρ range 0.106914. 927 oscillators lie within the archived locking range 0.01 of the median, and 73 lie outside it (0 in the low-frequency 5%, 23 in the interior, 50 in the high-frequency 5%). The lowest frequency among those outside points is ω=1.1712. The minimum is at ω=1.1704 and the maximum at ω=1.2000.

The W3 label is Incoherence. Near-zero fraction 0.0000 (Death requires 1), ρ range 0.193888 (Locking requires range ≤ 0.01), longest consecutive near-zero run 0 (Partial Death requires 15), and 0 validated nonzero plateau(s) (Partial Locking requires one). Sigma_hat is 3.364e-09.

Method C: 137 segments, 0 constant and 137 linear, 0 validated nonzero plateaus, plateau coverage 0.000. Largest constant candidate: none. Reading of the segment record: 3 linear segment(s) meet the Rule 3 length, relative-span, and rho-span tests but are not constant candidates; the longest has length 45, relative omega span 0.044, rho span 0.0000, mean 0.4151, omega [1.0979, 1.1155]; sigma_hat=3.364e-09, so these linear pieces partition a flat majority; that is the noise-floor behavior already seen on Rule 7 Locking profiles, not a separate missed plateau. No window contains a validated nonzero plateau.

Window changes: median(D)=0.001596, max(D)=0.006022, median(F)=0.001673, max(F)=0.009174. States W1/W2/W3 = Incoherence / Incoherence / Incoherence. Ranges = 0.194227, 0.194633, 0.193888. Validated plateau counts = 0, 0, 0. Plateau coverages = 0.000, 0.000, 0.000. Archived temporal review flag: clear. Agreement of the three window labels and a small pointwise change do not prove that the trajectory has converged.

Diagnostic tags, which are not new states: Flat majority with high-frequency tail.

W3 column for this δ, restricted to K≤1: Incoherence [0.00, 0.60] | Locking [0.65, 0.65] | Incoherence [0.70, 0.75] | Death [0.80, 1.00].

Relative to K=0.70, the full range grows from 0.015726 to 0.193888 and the central-90% range grows from 0.001093 to 0.106914. Oscillators outside the median by more than 0.01 increase from 1 to 73. 927 oscillators remain within 0.01 of the median. The outside points begin at ω=1.1712 and include the entire high-frequency 5%. Method C still returns no constant segment: the flat majority is split into linear pieces at a sigma_hat near 10⁻⁹, the same partition seen on the Locking control. The new structure is the high-frequency tail, not a missed interior plateau. Validated nonzero plateaus remain 0. The three windows keep the same state, so the tail is not a one-window fluctuation.

### Case E1: additional strict candidate δ=0.20, K=0.60

Control type on the named list, where applicable: pre-locking control. Strict candidate: yes. Near-boundary flag: no. K_H^(7)=0.55435, K−K_H^(7)=+0.04565, K_H^(6) difference=+0.03095, truncation K_H^(6)−K_H^(7)=+0.01470, 1−δ=0.800, K_exist=0.7902.

W3 ρ lies in [0.7256, 0.8190], range 0.093474, mean 0.7277, median 0.7256, near-zero fraction 0.0000. Pearson correlation of ρ with ω is 0.296 and the least-squares slope is 0.0308 per unit frequency. The 5–95% ρ width is 7.574e-05. The central 90% of oscillators in frequency order have ρ range 7.211e-05. 966 oscillators lie within the archived locking range 0.01 of the median, and 34 lie outside it (0 in the low-frequency 5%, 0 in the interior, 34 in the high-frequency 5%). The lowest frequency among those outside points is ω=1.1868. The minimum is at ω=1.0955 and the maximum at ω=1.2000.

The W3 label is Incoherence. Near-zero fraction 0.0000 (Death requires 1), ρ range 0.093474 (Locking requires range ≤ 0.01), longest consecutive near-zero run 0 (Partial Death requires 15), and 0 validated nonzero plateau(s) (Partial Locking requires one). Sigma_hat is 1.427e-10.

Method C: 58 segments, 0 constant and 58 linear, 0 validated nonzero plateaus, plateau coverage 0.000. Largest constant candidate: none. Reading of the segment record: 16 linear segment(s) meet the Rule 3 length, relative-span, and rho-span tests but are not constant candidates; the longest has length 133, relative omega span 0.132, rho span 0.0000, mean 0.7256, omega [0.8981, 0.9510]; sigma_hat=1.427e-10, so these linear pieces partition a flat majority; that is the noise-floor behavior already seen on Rule 7 Locking profiles, not a separate missed plateau. No window contains a validated nonzero plateau.

Window changes: median(D)=1.605e-05, max(D)=0.002838, median(F)=9.592e-06, max(F)=0.002954. States W1/W2/W3 = Incoherence / Incoherence / Incoherence. Ranges = 0.094232, 0.094142, 0.093474. Validated plateau counts = 0, 0, 0. Plateau coverages = 0.000, 0.000, 0.000. Archived temporal review flag: clear. Agreement of the three window labels and a small pointwise change do not prove that the trajectory has converged.

Diagnostic tags, which are not new states: Near-global-locking.

W3 column for this δ, restricted to K≤1: Incoherence [0.00, 0.60] | Locking [0.65, 0.65] | Incoherence [0.70, 0.75] | Death [0.80, 1.00].

These additional cells were chosen from the strict set after Cases A, C, and D were reserved. The sort prefers a larger excess over Eq. (7) and then a different diagnostic tag. They are included because their excess or their segment record differs from the reserved cases, not because they confirm one explanation.

## 8. Interpretation and limitations

There are 4 strict Incoherence candidates. They lie at δ=0.150, K from 0.40 to 0.40 (1 cell); δ=0.200, K from 0.60 to 0.75 (3 cells). 2 of them have a bulk rotation number inside the archived locking range while the full range exceeds 0.01. Possible missed local plateaus among the strict candidates: 1. The δ=0.20 Locking-to-Incoherence return is the structural change summarized above: Along δ=0.20 the W3 column on K≤1 is Incoherence [0.00, 0.60] | Locking [0.65, 0.65] | Incoherence [0.70, 0.75] | Death [0.80, 1.00]. K=0.65 is Locking with ρ range 7.510e-05. K=0.60 has 34 oscillators outside the median, all at high frequency, with range 0.093474. K=0.70 has ρ range 0.015726, central-90% range 0.001093, and 1 oscillator outside the median, at ω=1.2000. K=0.75 has ρ range 0.193888, central-90% range 0.106914, and 73 oscillators outside the median, the lowest of them at ω=1.1712. The return at K=0.70 is that one high-frequency oscillator. K=0.75 keeps a flat majority and replaces the single oscillator with a high-frequency tail. Neither profile is the frequency-tracking shape seen at δ=0.15, K=0.40. Window labels agree on 3 of 4 strict candidates; the temporally evolving tag is present on 1. Agreement supports treating the W3 label as stable across these three windows. It does not prove convergence.

The operational Incoherence label is the residual class after Death, Locking, Partial Death, and Partial Locking. The theoretical incoherent state is the stationary uniform density, which Eq. (7) and Eq. (6) say is linearly unstable once K passes the Hopf value, provided that value still lies below 1−δ. Those are different statements. On this grid the operational label does not always mean that the theoretical incoherent state is the object being observed: cells with K>K_H^(7) are past the printed linear-instability threshold, and the near-global-locking cells are finite-time profiles whose bulk frequency is already common. That fact does not by itself convict the classifier. Rule 7 uses the full range, the short deviant piece fails the membership rule, and the near-boundary cell at δ=0.15 is inside the truncation gap of Eq. (7). No strict candidate is declared a misclassification in this report.

Points that still need more than the three rotation-number windows:

- δ=0.15, K=0.40: the distance to the Hopf boundary is inside the Eq. (7) truncation, so this grid cannot place the crossing; the windows disagree at the archived review-flag level or the pointwise change exceeds the locking range in the median as well as the maximum; W3 contains a linear segment that meets Rule 3 geometry; one trajectory cannot show that the shelf is a stable plateau.
- δ=0.20, K=0.60: the three windows already agree on a flat majority plus a high-frequency tail; a longer copy of this trajectory is less useful than a finer K grid.
- δ=0.20, K=0.65: the three windows already agree on global locking; new data are not required to describe this control.
- δ=0.20, K=0.70: the three windows already agree on a flat majority plus a high-frequency tail; a longer copy of this trajectory is less useful than a finer K grid.
- δ=0.20, K=0.75: the three windows already agree on a flat majority plus a high-frequency tail; a longer copy of this trajectory is less useful than a finer K grid.

Limitations that this run does not remove: the O(δ⁷) remainder, the 0.05 coupling grid, N=1000 against the continuum, one shared initial phase, the missing self-coupling term already noted in the Eq. (7) validation (a factor (N−1)/N), and the absence of a finite-N eigenvalue. The death boundary is an existence threshold, not a statement that every initial condition reaches it. Eq. (5) is the incoherent-branch edge, not a second Hopf curve.

## 9. Recommended next experiment

The most useful next experiment is a finer K scan at δ=0.20, from just below the Eq. (6) root (K_H^(6)=0.56905) through the death onset at K_exist=0.7902, using the same N, windows, and unchanged Classifier v1. On the stored 0.05 grid the high-frequency tail has 34 oscillators at K=0.60, none at the single Locking cell K=0.65, 1 oscillator at K=0.70, and 73 oscillators at K=0.75. The scan measures how wide the locking interval is and how the tail grows. It does not retune the 0.01 rule. The δ=0.15, K=0.40 cell is a separate check: it lies on the Eq. (6) root, and its only window disagreement is one validated plateau of about 2% coverage in W1. A second initial phase at that single point can test the shelf. A linear-stability calculation is still required before either label is identified with the Hopf eigenvalue.
