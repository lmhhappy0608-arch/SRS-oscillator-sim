"""Stable identities for parameter points, realizations, and runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


def stable_id(kind: str, payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(f"{kind}\n{raw}".encode("utf-8")).hexdigest()
    return digest[:32]


@dataclass(frozen=True)
class RunSpec:
    """One planned simulation.

    ``run_id`` identifies the saved product (parameters, randomness, solver,
    and time horizon). ``lineage_id`` identifies the trajectory family shared
    by a later continuation. Operational fields such as the checkpoint
    interval and ``parent_run_id`` are recorded, but they are not part of the id.
    """

    model_version: str
    coupling_normalization: str
    pulse_function: str
    phase_response: str
    adjacency_convention: str
    frequency_rule: str
    frequency_pairing_rule: str
    K: float
    p: float
    Delta: float
    k_units: int
    p_units: int
    delta_units: int
    N: int
    frequency_seed: int | None
    network_seed: int
    initial_phase_seed: int
    network_realization_id: int
    initial_condition_id: int
    trial_id: int
    randomness_mode: str
    paired_group_id: str
    network_pairing_rule: str
    phase_pairing_rule: str
    seed_recipe: str
    solver: str
    solver_source: str
    dt: float
    rtol: float | None
    atol: float | None
    t_origin: float
    t_end: float
    observation_windows: tuple[tuple[float, float], ...]
    sample_order_parameter: bool
    order_parameter_dt: float | None
    checkpoint_interval: float
    parent_run_id: str | None
    master_seed: int

    def identity_payload(self) -> dict:
        return {
            "model_version": self.model_version,
            "coupling_normalization": self.coupling_normalization,
            "pulse_function": self.pulse_function,
            "phase_response": self.phase_response,
            "adjacency_convention": self.adjacency_convention,
            "frequency_rule": self.frequency_rule,
            "frequency_pairing_rule": self.frequency_pairing_rule,
            "k_units": self.k_units,
            "p_units": self.p_units,
            "delta_units": self.delta_units,
            "N": self.N,
            "frequency_seed": self.frequency_seed,
            "network_seed": self.network_seed,
            "initial_phase_seed": self.initial_phase_seed,
            "network_realization_id": self.network_realization_id,
            "initial_condition_id": self.initial_condition_id,
            "trial_id": self.trial_id,
            "randomness_mode": self.randomness_mode,
            "paired_group_id": self.paired_group_id,
            "network_pairing_rule": self.network_pairing_rule,
            "phase_pairing_rule": self.phase_pairing_rule,
            "seed_recipe": self.seed_recipe,
            "solver": self.solver,
            "solver_source": self.solver_source,
            "dt_hex": float(self.dt).hex(),
            "rtol_hex": None if self.rtol is None else float(self.rtol).hex(),
            "atol_hex": None if self.atol is None else float(self.atol).hex(),
            "t_origin_hex": float(self.t_origin).hex(),
            "t_end_hex": float(self.t_end).hex(),
            "windows_hex": [
                [float(start).hex(), float(end).hex()]
                for start, end in self.observation_windows
            ],
            "sample_order_parameter": bool(self.sample_order_parameter),
            "order_parameter_dt_hex": (
                None
                if self.order_parameter_dt is None
                else float(self.order_parameter_dt).hex()
            ),
        }

    def lineage_payload(self) -> dict:
        payload = self.identity_payload()
        for key in (
            "t_end_hex",
            "windows_hex",
            "sample_order_parameter",
            "order_parameter_dt_hex",
        ):
            payload.pop(key)
        return payload

    @property
    def run_id(self) -> str:
        return stable_id("run", self.identity_payload())

    @property
    def lineage_id(self) -> str:
        return stable_id("lineage", self.lineage_payload())
