# Eq. (7) incoherence Hopf boundary

Comparison of the Ariaratnam–Strogatz Hopf truncation with Classifier v1 labels on the saved p=1 grid. The linear-instability threshold, the finite-time state change, and the operational label are reported separately.

## A. Data and method

The index `stage1_runs/index.sqlite` has 6964 rows and 6962 completed runs. The completed p=1 slice has 1741 profiles. Every one has N=1000 and trial 0. Coupling runs from 0 to 3 in steps of 0.05 (61 values) and the frequency half-width from 0 to 1 in steps of 0.025 (41 values). The rectangle is incomplete: 38 widths stop at K=2 (41 values each), and only δ∈{0.05, 0.20, 0.80} continue through K=3 (61 values). The 760 missing cells are left empty.

One provenance group covers all 1741 rows: RK4 with dt=0.1, t from 0 to 14000, `master_seed=0`, `seed_recipe=sha256-v1`, paired randomness, `network_seed=8250318217634423533`, `initial_phase_seed=1364563909611740937`, null `frequency_seed`, and frequencies `1−δ … 1+δ` on a deterministic linspace. Network realization 0 and initial condition 0 are shared. The p=1 graph has a zero diagonal and 499500 upper-triangle edges. The adjacency hash regenerated from that network seed matches the stored hash of the sampled metadata file. The ODE is scaled by K/N and does not include the self-coupling term present in the paper sum, a relative factor (N−1)/N=0.999. That factor was not measured as a shift of the boundary.

Labels were recomputed in memory with archived Classifier v1.0.0-pilot (`stage2_classifier/classifier_v1`). They were not written back. W3=[12000, 14000] assigns the state. Near-zero is |ρ|<0.001. Death requires near-zero fraction exactly 1. Locking requires the whole-profile ρ range ≤0.01. Partial Death requires a consecutive near-zero run of at least 15 after those two rules. Partial Locking requires at least one Method C segment that passes Rule 3 with |mean ρ|≥0.001. The priority is Death, Locking, Partial Death, Partial Locking, Incoherence. Counts on the 1741 profiles: Locking 88, Death 975, Incoherence 329, Partial Locking 1, Partial Death 348.

The printed Eq. (7) in Ariaratnam and Strogatz (Phys. Rev. Lett. 86, 4278, 2001; arXiv:nlin/0012017) is

κ = (8γ/π) [ 1 + 16γ²/π² + 16(π²+80)γ⁴/π⁴ ] + O(γ⁷),

with γ the half-width of the uniform density on [1−γ, 1+γ] and κ the coupling. Here γ is δ and κ is K. The additive expression 8δ/π + 16δ²/π² + 16(π²+80)δ⁴/π⁴ drops the outer factor 8δ/π from the second and third terms, so it is not the printed truncation. The comparison below uses the printed truncation and does not refit any coefficient. At δ=0.10 the two expressions differ by 0.013; at δ=0.20 they differ by 0.043. δ=0 is degenerate (every frequency equals 1) and is not scored. The main window 0<δ≤0.20 was fixed before looking at the exits. A truncation value is inside the incoherent branch only when K_H < 1−δ. The series crosses that edge at δ=0.252.

Eq. (6) was solved only at δ∈{0.05, 0.10, 0.15, 0.20, 0.22, 0.225, 0.25}. On the imaginary axis the integrand is purely imaginary except for the Landau pole ω_p=√(K²+Ω²). The real part reduces to a cubic for v=Ω/ω_p. The imaginary part is a subtracted principal-value integral. A root is kept only when it is unique, Ω≠0, and K<1−δ. A direct split quadrature at λ=10⁻⁵+iΩ is a residual check. These roots diagnose the truncation. They are not the comparison baseline, and none was adjusted to the labels.

K_exit is defined only when the column begins with Incoherence and every cell after the first non-Incoherence cell stays non-Incoherence. A return to Incoherence leaves K_exit undefined. The first non-Incoherence label is not treated as a Hopf bifurcation. ΔK = K_exit − K_H^(7) is computed only for those sustained exits. No critical value was interpolated.

## B. Theoretical Hopf boundary

Printed truncation on the main window, all with K_H < 1−δ:

| δ | K_H^(7) | 1−δ |
| ---: | ---: | ---: |
| 0.025 | 0.0637 | 0.975 |
| 0.050 | 0.1279 | 0.950 |
| 0.075 | 0.1928 | 0.925 |
| 0.100 | 0.2592 | 0.900 |
| 0.125 | 0.3275 | 0.875 |
| 0.150 | 0.3988 | 0.850 |
| 0.175 | 0.4739 | 0.825 |
| 0.200 | 0.5544 | 0.800 |

Eq. (6) has one valid root at each solved width through δ=0.22. Principal-value imaginary residuals are at most 1.3×10⁻¹³, and the real residue condition holds to 10⁻¹⁵. Shifting the pole fraction by 0.001 changes K by less than 5×10⁻⁵ at δ≤0.20. The complex quadrature reproduces K to about 10⁻⁵, the size of the 10⁻⁵ smoothing, with an imaginary residual from 10⁻⁸ to 3×10⁻⁶.

| δ | Eq. (6) K | K_H^(7) | Eq. (6) − series |
| ---: | ---: | ---: | ---: |
| 0.050 | 0.12785 | 0.12785 | +0.00000 |
| 0.100 | 0.25920 | 0.25915 | +0.00005 |
| 0.150 | 0.39991 | 0.39876 | +0.00115 |
| 0.200 | 0.56905 | 0.55435 | +0.01470 |
| 0.220 | 0.67792 | 0.62355 | +0.05437 |
| 0.225 | no valid root | 0.64166 | — |
| 0.250 | no valid root | 0.73783 | — |

The leading piece 8δ/π is the Landau contribution for a pole at ω=1, so the normalization of the supplied Eq. (6) matches the printed leading coefficient. The numerical root tracks the full printed series through δ=0.10 and differs by 0.3% at δ=0.15. At δ=0.20 the series is 2.6% below the root (K≈0.57, inside the paper's stated 4% band for κ≤0.6). At δ=0.22 the series is low by 0.054 and the root is already more sensitive (dK/dz≈−0.23). At δ=0.225 and δ=0.25 the imaginary residual does not change sign anywhere inside K<1−δ. That absence is a property of the scanned incoherent branch, not an unstable root used as truth. The series at δ=0.25 is still formally below 1−δ (0.738<0.750) but is no longer a root of Eq. (6). From δ=0.275 onward the series itself lies above 1−δ.

## C. Observed transitions

δ=0 starts in Locking on [0, 0.75] and is Death from 0.80. It has no initial Incoherence interval.

On the eight main widths the initial state is Incoherence. Seven leave that interval permanently, and the next state is Locking in every case. Six of those seven place K_H^(7) inside (K_prev, K_exit]. Every defined ΔK is positive. The median is +0.026, about half of the 0.05 grid step; the minimum is +0.007. Using the upper end of the bracketing interval as K_exit produces a positive difference whenever the threshold lies in the gap, so this sign is the grid convention. It is not, by itself, evidence of an extra physical delay.

| δ | K_H^(7) | observed interval | before | after | K_exit | ΔK | status |
| ---: | ---: | --- | --- | --- | ---: | ---: | --- |
| 0.025 | 0.0637 | (0.05, 0.10] | Incoherence | Locking | 0.10 | +0.0363 | sustained; theory inside the gap |
| 0.050 | 0.1279 | (0.10, 0.15] | Incoherence | Locking | 0.15 | +0.0221 | sustained; theory inside the gap |
| 0.075 | 0.1928 | (0.15, 0.20] | Incoherence | Locking | 0.20 | +0.0072 | sustained; theory inside the gap |
| 0.100 | 0.2592 | (0.25, 0.30] | Incoherence | Locking | 0.30 | +0.0408 | sustained; theory inside the gap |
| 0.125 | 0.3275 | (0.30, 0.35] | Incoherence | Locking | 0.35 | +0.0225 | sustained; theory inside the gap |
| 0.150 | 0.3988 | (0.40, 0.45] | Incoherence | Locking | 0.45 | +0.0512 | sustained; theory 0.0012 below K_prev |
| 0.175 | 0.4739 | (0.45, 0.50] | Incoherence | Locking | 0.50 | +0.0261 | sustained; theory inside the gap |
| 0.200 | 0.5544 | first leave (0.60, 0.65], then Incoherence again | Incoherence | Locking, not sustained | — | — | non-monotonic; K_exit undefined |

The δ=0.20 column is Incoherence on [0.00, 0.60], Locking at the single point K=0.65, Incoherence on [0.70, 0.75], and Death from 0.80. No sustained exit is assigned, and no ΔK is plotted.

For δ>0.20 the same exit rule still produces a sustained label change, but those widths are not in the agreement count. δ=0.225 goes Incoherence through 0.70, Partial Locking at the single point 0.75, and Death from 0.80. The printed K_H=0.642 still satisfies K_H<1−δ, while Eq. (6) no longer has a valid root. From δ=0.250 the first sustained non-Incoherence label is Partial Death, followed later by Death. At δ=0.250 the series is still just inside the branch and is not a root of Eq. (6). From δ=0.275 the series is outside K<1−δ, and the observed incoherence exit continues to move down with 1−δ rather than with the growing truncation.

## D. Rotation-number evidence

The cells below are the archived W3 diagnostics. No new plateau rule was introduced.

Just below and just above three sustained Locking exits:

| δ | K | state | ρ range | near-zero fraction | mean ρ | validated nonzero plateaus |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0.050 | 0.10 | Incoherence | 0.1004 | 0 | 0.995 | 0 |
| 0.050 | 0.15 | Locking | 1.7×10⁻⁵ | 0 | 0.986 | 0 |
| 0.100 | 0.25 | Incoherence | 0.2066 | 0 | 0.968 | 0 |
| 0.100 | 0.30 | Locking | 2.5×10⁻⁴ | 0 | 0.943 | 0 |
| 0.150 | 0.40 | Incoherence | 0.3255 | 0 | 0.916 | 0 |
| 0.150 | 0.45 | Locking | 2.8×10⁻⁴ | 0 | 0.862 | 0 |

Each Locking label is the global-range rule: the range is far below 0.01, the near-zero fraction is 0, and the common value is well away from zero. Method C does not supply the label. On these flat profiles it returns only linear segments (10, 38, and 85 segments respectively), because the noise estimate sits at or near its floor and a microscopic slope is fit as linear. The largest segment ρ span is 10⁻⁵. The collective-frequency evidence is the global range itself, not a Method C plateau.

At the single Partial Locking cell, δ=0.225 and K=0.75, the global range is 0.768 and the near-zero fraction is 0, so the locking rule does not fire. Method C reports one validated nonzero plateau: 25 oscillators, coverage 0.025, level 0.5665. The neighboring cells are Incoherence at K=0.70 (range 0.669, no plateau) and Death at K=0.80.

Along δ=0.20 the ranges are 0.484 at K=0.55, 0.093 at K=0.60, 7.5×10⁻⁵ at K=0.65, 0.0157 at K=0.70, and 0.194 at K=0.75. Near-zero fraction is 0 until Death at 0.80. The K=0.65 label is again the global-range rule, at mean ρ=0.647, with 43 linear segments and no validated plateau. At K=0.70 the whole range 0.0157 sits in one Method C segment of length 4; the profile is otherwise much tighter, and the Incoherence label is the range threshold 0.01. At K=0.75 the range has reopened to 0.194. None of these profiles has a validated nonzero plateau.

A flat nonzero rotation number supports a collective-frequency reading. It does not show that an eigenvalue crossed the imaginary axis. A missing Method C plateau does not remove the global-range evidence, and a broad range does not prove that the incoherent state is still linearly stable.

## E. Disagreements and limitations

**δ=0.15, one grid step plus 0.001.** Direct evidence: Incoherence through K=0.40 (range 0.325, near-zero fraction 0) and Locking at K=0.45 (range 2.8×10⁻⁴, mean ρ=0.862). K_H^(7)=0.3988 and the Eq. (6) root 0.3999 both sit just below K=0.40. The state change is the same Locking transition as the other main widths, one stored step later than the bracketing interval. The additive transcription would have fallen inside (0.40, 0.45]; the miss is tied to using the printed series, and the Eq. (6) root does not remove it. Nothing in the rotation numbers identifies which side of 0.40 the continuum threshold occupies.

**δ=0.20, label return.** Direct evidence: the sequence in section C, the ranges in section D, and a valid Eq. (6) root at 0.569 against a series value 0.554. Both lie in a region that is still labeled Incoherence, with range 0.484 at K=0.55. Locking occupies only K=0.65. The return at K=0.70 is the 0.01 range rule, carried by a length-4 segment, while K=0.75 is a wider spread (0.194) before Death at 0.80. K_exit is undefined. The truncation gap of 0.015 does not close the distance from 0.55 to the single Locking cell. A possible reading is that death at K=0.80, finite N, one initial phase, and the range threshold all sit close enough to interfere. That reading is not a measurement of the eigenvalue.

**δ=0.225, one Partial Locking cell, exploratory.** Direct evidence: the only Partial Locking label in the p=1 slice, produced by one Method C plateau of 25 oscillators, between Incoherence and Death. Eq. (6) has no valid root inside K<1−δ, while the series still prints 0.642. This width is outside the main window and is not counted as agreement or disagreement with Eq. (7).

**δ≥0.25, another boundary.** Direct evidence: the sustained exit is Partial Death, then Death. From δ=0.275 the printed series is above 1−δ, so it is not an incoherent-branch threshold. The earlier Eq. (5) comparison associated this Partial Death onset with low-frequency stopping near K=1−δ. The earlier death-boundary comparison associated the later Death onset with a saddle-node for δ≤0.275 and an end-point for δ≥0.300, with the existence mechanism changing at δ_*=0.296. The label route seen here changes earlier, near δ=0.20–0.25. Those are three different curves. Small δ can lock at the Hopf scale and reach Death while K is still below 1−δ, because the death threshold stays near 0.80 and 1−δ is higher. Larger δ meets K=1−δ first, so the observed exit from Incoherence is Partial Death rather than a locking transition.

The grid, the O(δ⁷) remainder, the missing self-coupling, N=1000 against the continuum, the single shared initial phase, and the 0.01 locking rule are all unresolved contributors. No ensemble was run. No eigenvalue of the finite-N system was computed.

## F. Conclusions

1. On the pre-set window, the printed Eq. (7) truncation lies in the incoherent branch at all eight widths. Seven have a sustained Incoherence→Locking exit. The truncation falls inside the adjacent K interval for six of those seven. The seventh, δ=0.15, misses that interval by 0.0012, one stored step above the prediction. δ=0.20 has no sustained exit. The positive ΔK values are what the upper endpoint of a 0.05 gap produces; their median, +0.026, matches that grid scale. This is agreement of a label boundary with a small-δ truncation, within about one grid step, on six widths.

2. Classifier v1 does record a collective-frequency change at those exits. The Locking label is the global ρ range, and the common value is nonzero (about 0.86 to 0.99 on the illustrated cells) with near-zero fraction 0. Method C plateaus do not make these Locking calls. The one Partial Locking call in the whole slice is exploratory and is a Method C plateau.

3. The only non-monotonic main-window column is δ=0.20: Incoherence, one Locking cell, Incoherence, Death. δ=0 has no incoherence interval. No other main width returns to Incoherence after leaving it.

4. The rotation-number plateaus and ranges are indirect consistency evidence. They do not prove that a discrete eigenvalue crossed the imaginary axis, and the absence of a plateau does not prove stability. Eq. (6) supports the printed series as an approximation through δ≈0.15 and shows a 2.6% truncation gap at δ=0.20. It does not turn the classifier labels into a stability proof. The Eq. (5) and death-boundary comparisons explain why locking and partial death compete at larger δ; they were not rerun here.

5. A finer K grid is warranted at δ=0.15, where the prediction sits 0.001 below a stored point, and at δ=0.20, where locking occupies one cell and the next cell is a 0.01-threshold return. A linear stability calculation would be required before identifying any of these label changes with the Hopf bifurcation. Neither was started.
