"""Deterministic CPU-only synthetic seam-tracking and DAgger-style demo.

This toy task studies lateral tracking error on a generated curve. It says
nothing about weld formation, weld quality, or a physical robot.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Iterable

from .plugins import Supervisor, load_supervisor


@dataclass(frozen=True)
class SyntheticSeam:
    """A procedural 2-D path, expressed as lateral position by sample index."""

    offset_mm: float
    slope_mm: float
    curvature_mm: float
    wave_mm: float
    phase_rad: float

    def at(self, sample: int) -> float:
        t = float(sample)
        return (
            self.offset_mm
            + self.slope_mm * t
            + self.curvature_mm * t * (t - 1.0) / 2.0
            + self.wave_mm * math.sin(0.19 * t + self.phase_rad)
        )


def make_seam(rng: random.Random) -> SyntheticSeam:
    """Generate one reproducible, geometry-free synthetic seam profile."""

    return SyntheticSeam(
        offset_mm=rng.uniform(-1.5, 1.5),
        slope_mm=rng.uniform(-0.035, 0.035),
        curvature_mm=rng.uniform(-0.001, 0.001),
        wave_mm=rng.uniform(0.0, 0.04),
        phase_rad=rng.uniform(-math.pi, math.pi),
    )


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass
class LinearCorrectionPolicy:
    """A single-gain controller fitted from synthetic critic corrections."""

    gain: float = 0.0
    max_step_mm: float = 0.25

    def act(self, lateral_error_mm: float) -> float:
        return clamp(self.gain * lateral_error_mm, self.max_step_mm)

    def fit(self, labels: Iterable[tuple[float, float]]) -> None:
        """Fit a through-origin least-squares gain to aggregated corrections."""

        numerator = 0.0
        denominator = 0.0
        for error, correction in labels:
            numerator += error * correction
            denominator += error * error
        if denominator:
            self.gain = max(0.0, min(1.0, numerator / denominator))


@dataclass(frozen=True)
class SyntheticCritic:
    """A deterministic threshold rule, not an LLM or a human reviewer."""

    review_threshold_mm: float = 0.45

    def review(self, lateral_error_mm: float) -> float | None:
        if abs(lateral_error_mm) < self.review_threshold_mm:
            return None
        # Return the desired correction as a training label. The simulated
        # actuator separately clips the applied correction to its step limit.
        return lateral_error_mm


def collect_dagger_data(
    *,
    seed: int,
    rounds: int,
    episodes_per_round: int,
    steps: int,
    supervisor: Supervisor | None = None,
) -> tuple[LinearCorrectionPolicy, int, int]:
    """Collect intervention labels, aggregate them, and refit each round."""

    rng = random.Random(seed)
    policy = LinearCorrectionPolicy()
    critic: Supervisor = supervisor or SyntheticCritic()
    labels: list[tuple[float, float]] = []
    interventions = 0

    for _round in range(rounds):
        for _episode in range(episodes_per_round):
            seam = make_seam(rng)
            position_mm = 0.0
            for sample in range(steps):
                error = seam.at(sample) - position_mm
                correction = policy.act(error)
                expert_correction = critic.review(error)
                if expert_correction is not None:
                    # The threshold critic supplies a synthetic expert label
                    # and intervenes in this rollout. This is not an actual
                    # external supervisor review.
                    labels.append((error, expert_correction))
                    correction = clamp(expert_correction, policy.max_step_mm)
                    interventions += 1
                position_mm += correction
        policy.fit(labels)

    return policy, interventions, len(labels)


def tracking_mean_abs_error(
    *, seed: int, gain: float, episodes: int, steps: int, max_step_mm: float = 0.25
) -> float:
    """Measure only synthetic lateral tracking error over generated paths."""

    rng = random.Random(seed)
    total = 0.0
    count = 0
    for _ in range(episodes):
        seam = make_seam(rng)
        position_mm = 0.0
        for sample in range(steps):
            error = seam.at(sample) - position_mm
            total += abs(error)
            count += 1
            position_mm += clamp(gain * error, max_step_mm)
    return total / count if count else 0.0


def run_demo(
    *,
    seed: int = 7,
    rounds: int = 5,
    episodes_per_round: int = 8,
    steps: int = 48,
    supervisor: Supervisor | None = None,
) -> dict[str, int | float | str]:
    policy, interventions, labels = collect_dagger_data(
        seed=seed,
        rounds=rounds,
        episodes_per_round=episodes_per_round,
        steps=steps,
        supervisor=supervisor,
    )
    evaluation_seed = seed + 1_000
    baseline = tracking_mean_abs_error(
        seed=evaluation_seed, gain=0.0, episodes=24, steps=steps
    )
    refined = tracking_mean_abs_error(
        seed=evaluation_seed, gain=policy.gain, episodes=24, steps=steps
    )
    expert_reference = tracking_mean_abs_error(
        seed=evaluation_seed, gain=1.0, episodes=24, steps=steps
    )
    using_synthetic_critic = supervisor is None
    return {
        "demo": "synthetic-cpu-seam-dagger",
        "seed": seed,
        "rounds": rounds,
        "dagger_labels": labels,
        "supervisor_mode": "synthetic-critic" if using_synthetic_critic else "configured-provider",
        "synthetic_critic_interventions": interventions if using_synthetic_critic else 0,
        "configured_provider_interventions": 0 if using_synthetic_critic else interventions,
        "learned_gain": round(policy.gain, 6),
        "synthetic_lateral_tracking_mean_abs_mm_before": round(baseline, 6),
        "synthetic_lateral_tracking_mean_abs_mm_after": round(refined, 6),
        "synthetic_expert_reference_mean_abs_mm": round(expert_reference, 6),
        "interpretation": "Synthetic lateral tracking only; not weld quality or robot validation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--episodes-per-round", type=int, default=8)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument(
        "--supervisor",
        choices=("synthetic", "provider"),
        default="synthetic",
        help="use the built-in synthetic critic or explicitly configured provider",
    )
    args = parser.parse_args()
    if min(args.rounds, args.episodes_per_round, args.steps) < 1:
        parser.error("rounds, episodes-per-round, and steps must all be positive")
    supervisor = None
    if args.supervisor == "provider":
        supervisor = load_supervisor()
        if supervisor is None:
            parser.error(
                "--supervisor provider requires ROBO_WELD_SUPERVISOR_PROVIDER=module:factory"
            )
    print(
        json.dumps(
            run_demo(
                seed=args.seed,
                rounds=args.rounds,
                episodes_per_round=args.episodes_per_round,
                steps=args.steps,
                supervisor=supervisor,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
