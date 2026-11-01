"""Parameter grids without cumulative floating-point drift."""

from __future__ import annotations

UNIT = 1_000_000_000


def quantize(value: float) -> int:
    """Map a parameter onto the nearest multiple of 1e-9."""
    return int(round(float(value) * UNIT))


def from_units(units: int) -> float:
    return int(units) / UNIT


def canonical_label(value: float) -> str:
    return format_units(quantize(value))


def format_units(units: int) -> str:
    sign = "-" if int(units) < 0 else ""
    whole, frac = divmod(abs(int(units)), UNIT)
    if frac == 0:
        return f"{sign}{whole}"
    digits = f"{frac:09d}".rstrip("0")
    return f"{sign}{whole}.{digits}"


def inclusive_range(vmin: float, vmax: float, step: float) -> tuple[float, ...]:
    """Inclusive arithmetic grid.

    Values are ``vmin + i * step`` in integer units. A range that does not
    land on ``vmax`` raises ``ValueError`` instead of silently dropping the
    endpoint or accumulating binary error.
    """
    start = quantize(vmin)
    stop = quantize(vmax)
    stride = quantize(step)
    if stride <= 0:
        raise ValueError(f"step must be positive, got {step}")
    if stop < start:
        raise ValueError(f"max {vmax} is below min {vmin}")
    span = stop - start
    if span % stride != 0:
        raise ValueError(
            "grid from "
            f"{canonical_label(vmin)} to {canonical_label(vmax)} "
            f"with step {canonical_label(step)} does not include the endpoint"
        )
    count = span // stride
    return tuple(from_units(start + index * stride) for index in range(count + 1))


def canonicalize_values(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    """Quantize an explicit value list and reject duplicates."""
    if not values:
        raise ValueError("value list is empty")
    seen: set[int] = set()
    ordered: list[float] = []
    for value in values:
        units = quantize(value)
        if units in seen:
            raise ValueError(f"duplicate grid value {canonical_label(value)}")
        seen.add(units)
        ordered.append(from_units(units))
    return tuple(ordered)
