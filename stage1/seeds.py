"""Reproducible seeds for networks, initial phases, and paired scans."""

from __future__ import annotations

import hashlib

SEED_RECIPE = "sha256-v1"
NETWORK_RULE_SHARED = "shared_upper_triangle_uniforms_thresholded_by_p"
NETWORK_RULE_INDEPENDENT = "independent_uniforms_per_parameter_cell"
PHASE_RULE_SHARED = "shared_initial_phases_across_parameters"
PHASE_RULE_INDEPENDENT = "independent_initial_phases_per_parameter_cell"
FREQUENCY_RULE = "deterministic_linspace_1_minus_delta_to_1_plus_delta"
FREQUENCY_PAIRING_RULE = "shared_quantile_index_across_delta"

_SEED_MASK = (1 << 63) - 1


def derive_seed(*parts: object) -> int:
    """Return a signed 63-bit seed.

    The recipe is the SHA-256 of a newline-joined payload. Only the first 8
    bytes are kept, with the high bit cleared so the value fits in SQLite's
    signed 64-bit integer.
    """
    payload = "\n".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") & _SEED_MASK


def assign_realizations(
    mode: str,
    trial_id: int,
    fixed_network_realization_id: int,
    fixed_initial_condition_id: int,
) -> tuple[int, int]:
    """Map a trial index onto network and initial-condition ids."""
    if mode == "paired":
        return trial_id, trial_id
    if mode == "network":
        return trial_id, fixed_initial_condition_id
    if mode == "initial_condition":
        return fixed_network_realization_id, trial_id
    if mode == "independent":
        return trial_id, trial_id
    raise ValueError(f"unknown randomness mode {mode!r}")


def network_and_phase_seeds(
    mode: str,
    master_seed: int,
    trial_id: int,
    network_realization_id: int,
    initial_condition_id: int,
    k_label: str,
    p_label: str,
    delta_label: str,
) -> tuple[int, int, str, str]:
    """Seeds and pairing-rule labels for one simulation.

    Paired, network, and initial-condition modes keep the network seed
    independent of ``K``, ``p``, and ``Delta``. Different ``p`` then threshold
    one shared uniform field, so the graphs are nested rather than identical.
    Independent mode mixes the parameter labels into both seeds.
    """
    if mode == "independent":
        network_seed = derive_seed(
            SEED_RECIPE,
            "network",
            "independent",
            master_seed,
            k_label,
            p_label,
            delta_label,
            "trial",
            trial_id,
        )
        phase_seed = derive_seed(
            SEED_RECIPE,
            "phase",
            "independent",
            master_seed,
            k_label,
            p_label,
            delta_label,
            "trial",
            trial_id,
        )
        return (
            network_seed,
            phase_seed,
            NETWORK_RULE_INDEPENDENT,
            PHASE_RULE_INDEPENDENT,
        )

    network_seed = derive_seed(
        SEED_RECIPE,
        "network",
        master_seed,
        "realization",
        network_realization_id,
    )
    phase_seed = derive_seed(
        SEED_RECIPE,
        "phase",
        master_seed,
        "realization",
        initial_condition_id,
    )
    return network_seed, phase_seed, NETWORK_RULE_SHARED, PHASE_RULE_SHARED


def paired_group_id(
    mode: str,
    network_realization_id: int,
    initial_condition_id: int,
    trial_id: int,
    k_label: str,
    p_label: str,
    delta_label: str,
) -> str:
    if mode == "paired":
        return f"paired-net-{network_realization_id}-ic-{initial_condition_id}"
    if mode == "network":
        return f"network-net-{network_realization_id}-ic-{initial_condition_id}"
    if mode == "initial_condition":
        return f"initial-net-{network_realization_id}-ic-{initial_condition_id}"
    if mode == "independent":
        return (
            f"independent-trial-{trial_id}-K-{k_label}-p-{p_label}-Delta-{delta_label}"
        )
    raise ValueError(f"unknown randomness mode {mode!r}")
