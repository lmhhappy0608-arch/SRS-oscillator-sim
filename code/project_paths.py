"""Project layout: Python scripts live in code/; data and outputs at repo root."""

from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
