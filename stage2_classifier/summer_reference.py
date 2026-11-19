"""Feature helpers for the unmodified summer classifier.

The state rule is ``classify_state`` in ``code/Ariaratnam2001.py``.
These two functions used to live beside Classifier v1. The cross-checks
still need them after that directory is removed.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from Ariaratnam2001 import (  # noqa: E402
    classify_state,
    find_locking_plateaus,
    has_consecutive_zero_rotation,
)


def summer_defaults() -> dict:
    defaults = {}
    for name, param in inspect.signature(classify_state).parameters.items():
        if param.default is not inspect.Parameter.empty:
            defaults[name] = param.default
    return defaults


def profile_features(omega: np.ndarray, rho: np.ndarray, defaults: dict) -> dict:
    plateaus = find_locking_plateaus(omega, rho)
    nonzero = [item for item in plateaus if abs(item["rho_center"]) >= defaults["tol_death"]]
    largest = max((item["fraction"] for item in nonzero), default=0.0)
    return {
        "zero_fraction": float(np.mean(np.abs(rho) < defaults["tol_death"])),
        "has_consecutive_zero": int(has_consecutive_zero_rotation(omega, rho)),
        "rho_range": float(rho.max() - rho.min()),
        "plateau_count": len(plateaus),
        "nonzero_plateau_count": len(nonzero),
        "largest_nonzero_fraction": float(largest),
        "identity_fraction": float(np.mean(np.abs(rho - omega) < defaults["tol_identity"])),
    }
