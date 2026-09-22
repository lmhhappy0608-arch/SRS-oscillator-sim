"""Paths for classification tools inside this repo.

Raw Winfree trials stay in raw_trials/ (not in git). Classification
code, figures, and tables live under classification/.
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
CLASSIFICATION_ROOT = CODE_DIR.parent
CURSOR_ROOT = CLASSIFICATION_ROOT.parent
CURSOR_CODE_DIR = CURSOR_ROOT / "code"
PROJECT_ROOT = CURSOR_ROOT

if str(CURSOR_CODE_DIR) not in sys.path:
    sys.path.append(str(CURSOR_CODE_DIR))

DEFAULT_RAW_DIR = CURSOR_ROOT / "raw_trials"
DEFAULT_MIXTURE_FIGURE_DIR = CLASSIFICATION_ROOT / "figures" / "rotation_mixture"
DEFAULT_TABLE_DIR = CLASSIFICATION_ROOT / "tables"
DEFAULT_RESIDUAL_FIGURE_DIR = CLASSIFICATION_ROOT / "figures" / "mixture_residuals"
DEFAULT_SIMPLEX_FIGURE_DIR = CLASSIFICATION_ROOT / "figures" / "mixture_simplex"
DEFAULT_PLATEAU_FIGURE_DIR = CLASSIFICATION_ROOT / "figures" / "plateau_observe"
DEFAULT_P_VALUES = (0.2, 0.4, 0.6, 0.8, 1.0)


def resolve_user_path(
    path: Path,
    classification_root: Path = CLASSIFICATION_ROOT,
    cursor_root: Path = CURSOR_ROOT,
) -> Path:
    """Resolve a CLI path against classification/, then the repo root."""
    path = Path(path)
    if path.is_absolute():
        return path
    under_classification = classification_root / path
    under_cursor = cursor_root / path
    if under_cursor.exists() and not under_classification.exists():
        return under_cursor
    return under_classification
