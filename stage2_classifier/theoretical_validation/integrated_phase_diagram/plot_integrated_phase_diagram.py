"""Overlay the archived Eq. (5), Eq. (7), and death-existence boundaries.

The state colors and the Hopf formula come from the Eq. (7) validation.
The death curve is the same existence calculation as the death-boundary
validation. Saddle-node and end-point are one boundary, drawn in one color
with two line styles. Nothing already archived is rewritten.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-integrated-phase")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "stage2_classifier" / "theoretical_validation" / "eq7_hopf"))
sys.path.insert(0, str(ROOT / "stage2_classifier" / "theoretical_validation" / "death_boundary"))

import validate_death_boundary as death  # noqa: E402
import validate_eq7 as hopf  # noqa: E402

FIGURE_PATH = HERE / "integrated_theoretical_phase_diagram.png"
MAIN_HOPF = 0.20
MISSING_COLOR = "#d0d0d0"
HOPF_COLOR = "#e66101"
# Teal, not the Locking blue. Solid and dashed mark the two mechanisms.
DEATH_COLOR = "#0b7d74"
DEATH_END_STYLE = (0, (10, 5))
NOTE = (
    "Eq. (7) is a small-δ approximation; Eq. (5) is branch-conditional; "
    "Eq. (8)–(9) gives equilibrium existence, not guaranteed attraction."
)


def hopf_curves() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Printed Eq. (7), solid on (0, 0.20] and dashed only while K < 1-delta."""
    samples = np.linspace(0.001, 0.40, 500)
    main_delta, main_k, extra_delta, extra_k = [], [], [], []
    for delta in samples:
        coupling = hopf.series_eq7(float(delta))
        if coupling >= 1.0 - float(delta):
            break
        if delta <= MAIN_HOPF:
            main_delta.append(float(delta))
            main_k.append(coupling)
        else:
            extra_delta.append(float(delta))
            extra_k.append(coupling)
    main_delta.append(MAIN_HOPF)
    main_k.append(hopf.series_eq7(MAIN_HOPF))
    if extra_delta:
        extra_delta.insert(0, MAIN_HOPF)
        extra_k.insert(0, hopf.series_eq7(MAIN_HOPF))
    return (
        np.array(main_delta),
        np.array(main_k),
        np.array(extra_delta),
        np.array(extra_k),
    )


def death_curves(switch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Saddle-node and end-point pieces of the archived existence boundary."""
    samples = np.linspace(0.0, 1.0, 201)
    saddle_delta, saddle_k, end_delta, end_k = [], [], [], []
    for delta in samples:
        item = death.existence_boundary(float(delta))
        if item["status"] != "ok":
            raise RuntimeError(f"death boundary unresolved at delta={delta}: {item}")
        expected = "saddle-node" if delta <= switch + 1e-9 else "end-point"
        if item["mechanism"] != expected and abs(delta - switch) > 0.01:
            raise RuntimeError(f"branch changed away from the switch at delta={delta}: {item['mechanism']}")
        if delta <= switch:
            saddle_delta.append(float(delta))
            saddle_k.append(float(item["K_exist"]))
        if delta >= switch:
            end_delta.append(float(delta))
            end_k.append(float(item["K_exist"]))
    joined = death.existence_boundary(float(switch))
    if joined["status"] != "ok":
        raise RuntimeError(joined)
    saddle_delta.append(float(switch))
    saddle_k.append(float(joined["K_exist"]))
    end_delta.insert(0, float(switch))
    end_k.insert(0, float(joined["K_exist"]))
    return (
        np.array(saddle_delta),
        np.array(saddle_k),
        np.array(end_delta),
        np.array(end_k),
    )


def state_field(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deltas = np.array(sorted({row["Delta"] for row in records}))
    ks = np.array(sorted({row["K"] for row in records}))
    delta_edges = np.concatenate(([deltas[0] - 0.0125], 0.5 * (deltas[:-1] + deltas[1:]), [deltas[-1] + 0.0125]))
    k_edges = np.concatenate(([ks[0] - 0.025], 0.5 * (ks[:-1] + ks[1:]), [ks[-1] + 0.025]))
    delta_index = {round(value, 10): index for index, value in enumerate(deltas)}
    k_index = {round(value, 10): index for index, value in enumerate(ks)}
    code = {state: index for index, state in enumerate(hopf.PHASE_STATES)}
    field = np.full((ks.size, deltas.size), np.nan)
    for row in records:
        field[k_index[round(row["K"], 10)], delta_index[round(row["Delta"], 10)]] = code[row["state"]]
    return delta_edges, k_edges, field


def draw_overlay(ax, curves: dict) -> None:
    ax.plot([0.0, 1.0], [1.0, 0.0], color="black", lw=1.5, ls=(0, (1.2, 1.4)), zorder=3)
    ax.plot(curves["hopf_main_d"], curves["hopf_main_k"], color=HOPF_COLOR, lw=2.2, zorder=4)
    ax.plot(curves["hopf_extra_d"], curves["hopf_extra_k"], color=HOPF_COLOR, lw=2.0, ls="--", zorder=4)
    ax.plot(curves["sn_d"], curves["sn_k"], color=DEATH_COLOR, lw=2.4, zorder=4)
    ax.plot(curves["ep_d"], curves["ep_k"], color=DEATH_COLOR, lw=2.4, ls=DEATH_END_STYLE, zorder=4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.30)
    ax.set_xlabel("Frequency half-width δ")
    ax.set_ylabel("Coupling K")


def draw_figure(records: list[dict], curves: dict) -> None:
    delta_edges, k_edges, field = state_field(records)
    cmap = ListedColormap(hopf.PHASE_COLORS)
    cmap.set_bad(MISSING_COLOR)
    norm = BoundaryNorm(np.arange(-0.5, len(hopf.PHASE_COLORS) + 0.5, 1), cmap.N)
    masked = np.ma.masked_invalid(field)

    fig, ax = plt.subplots(figsize=(8.4, 6.7))
    ax.pcolormesh(delta_edges, k_edges, masked, cmap=cmap, norm=norm, shading="flat", zorder=0)
    draw_overlay(ax, curves)
    ax.set_title("Zoomed phase diagram")

    handles = [
        Patch(facecolor=color, edgecolor="0.35", linewidth=0.3, label=state)
        for state, color in zip(hopf.PHASE_STATES, hopf.PHASE_COLORS)
    ]
    handles.extend(
        [
            Line2D([0], [0], color="black", lw=1.5, ls=(0, (1.2, 1.4)), label="Eq. (5): Incoherent-branch death onset"),
            Line2D([0], [0], color=HOPF_COLOR, lw=2.2, label="Eq. (7): Approximate Hopf boundary"),
            Line2D([0], [0], color=HOPF_COLOR, lw=2.0, ls="--", label="Eq. (7): exploratory extension"),
            Line2D([0], [0], color=DEATH_COLOR, lw=2.4, label="Death existence: Saddle-node"),
            Line2D([0], [0], color=DEATH_COLOR, lw=2.4, ls=DEATH_END_STYLE, label="Death existence: End-point"),
        ]
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle("Winfree model, p=1, N=1000, trial 0, W3=[12000, 14000], Classifier v1.0.0-pilot", fontsize=11)
    fig.text(0.5, 0.005, NOTE, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 0.94))
    fig.savefig(FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)


def check(records: list[dict], curves: dict) -> None:
    counts: dict[str, int] = {}
    for row in records:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    printed = hopf.series_eq7(0.10)
    factor = 1.0 + 16.0 * 0.01 / hopf.PI**2 + 16.0 * (hopf.PI**2 + 80.0) * 0.0001 / hopf.PI**4
    multiplicative = (8.0 * 0.10 / hopf.PI) * factor
    additive = 8.0 * 0.10 / hopf.PI + 16.0 * 0.01 / hopf.PI**2 + 16.0 * (hopf.PI**2 + 80.0) * 0.0001 / hopf.PI**4
    archived = {
        0.0: 0.7698,
        0.2: 0.7902,
        0.275: 0.8082,
        0.3: 0.8155,
        0.5: 0.8889,
        1.0: 1.1202,
    }
    print("profiles", len(records), "states", counts)
    print("missing_cells", 41 * 61 - len(records))
    print(
        "eq7_multiplicative",
        f"{printed:.5f}",
        "matches",
        abs(printed - multiplicative) < 1e-12,
        "differs_from_additive",
        f"{printed - additive:+.5f}",
    )
    print("hopf_main_end", f"{curves['hopf_main_d'][-1]:.3f}", f"{curves['hopf_main_k'][-1]:.4f}")
    print(
        "hopf_extra",
        f"{curves['hopf_extra_d'][0]:.3f}",
        f"{curves['hopf_extra_k'][0]:.4f}",
        "to",
        f"{curves['hopf_extra_d'][-1]:.3f}",
        f"{curves['hopf_extra_k'][-1]:.4f}",
        "below_branch",
        curves["hopf_extra_k"][-1] < 1.0 - curves["hopf_extra_d"][-1],
    )
    print("switch", f"{curves['switch']:.4f}")
    print(
        "death_join",
        f"{curves['sn_k'][-1]:.4f}",
        f"{curves['ep_k'][0]:.4f}",
        "gap",
        f"{abs(curves['sn_k'][-1] - curves['ep_k'][0]):.2e}",
    )
    for delta, expected in archived.items():
        item = death.existence_boundary(delta)
        print(
            f"death d={delta:.3f}",
            item["mechanism"],
            f"{item['K_exist']:.4f}",
            "archived",
            f"{expected:.4f}",
            "ok",
            abs(item["K_exist"] - expected) < 5e-4,
        )
    print("figure", FIGURE_PATH)


def main() -> None:
    switch = float(brentq(death.endpoint_slope, 0.2, 0.4))
    records, _meta = hopf.load_records()
    sn_d, sn_k, ep_d, ep_k = death_curves(switch)
    hopf_main_d, hopf_main_k, hopf_extra_d, hopf_extra_k = hopf_curves()
    curves = {
        "switch": switch,
        "sn_d": sn_d,
        "sn_k": sn_k,
        "ep_d": ep_d,
        "ep_k": ep_k,
        "hopf_main_d": hopf_main_d,
        "hopf_main_k": hopf_main_k,
        "hopf_extra_d": hopf_extra_d,
        "hopf_extra_k": hopf_extra_k,
    }
    draw_figure(records, curves)
    check(records, curves)


if __name__ == "__main__":
    main()
