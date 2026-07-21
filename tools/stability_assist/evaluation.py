from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

import numpy as np
import torch

from .environment import EnvironmentConfig, VectorizedQwasEnv, generate_held_button_sequence
from .model import Actor
from .spec import DEFAULT_PHYSICS, normalize_observation


def _actor_action(actor: Actor | None, raw_observation: np.ndarray, device: torch.device) -> np.ndarray:
    if actor is None:
        return np.zeros((raw_observation.shape[0], 4), dtype=np.float32)
    observation = torch.from_numpy(normalize_observation(raw_observation)).to(device)
    with torch.no_grad():
        return actor(observation).cpu().numpy().astype(np.float32)


def make_scenario(seed: int, duration: float, config: EnvironmentConfig) -> dict[str, Any]:
    initial_env = VectorizedQwasEnv(1, seed, config)
    rng = np.random.default_rng(seed + 100_003)
    max_steps = int(np.ceil(duration / config.dt_min)) + 2
    dt = rng.uniform(config.dt_min, config.dt_max, max_steps).astype(np.float32)
    buttons = generate_held_button_sequence(seed + 200_003, max_steps, dt,
                                            config.input_hold_min, config.input_hold_max)
    return {"snapshot": initial_env.snapshot(), "dt": dt, "buttons": buttons}


def run_scenario(actor: Actor | None, mode: str, scenario: dict[str, Any],
                 config: EnvironmentConfig, device: torch.device) -> dict[str, Any]:
    env = VectorizedQwasEnv(1, 0, config)
    env.load_snapshot(scenario["snapshot"])
    last_episode: dict[str, Any] | None = None
    for buttons, dt in zip(scenario["buttons"], scenario["dt"]):
        env.set_controls(buttons[None, :], float(dt))
        action = _actor_action(actor if mode == "trained" else None, env.raw_observation(), device)
        _, _, done, info = env.step(action, mode=mode, auto_reset=False, resample_controls=False)
        if done[0]:
            last_episode = info["episodes"][0]
            break
    if last_episode is None:
        steps = max(int(env.episode_steps[0]), 1)
        last_episode = {
            "survival_time": float(env.elapsed[0]), "crashed": False, "reason": "duration",
            "max_tilt_degrees": float(np.rad2deg(env.max_tilt[0])),
            "mean_pitch_roll_rate": float(env.rate_sum[0] / steps),
            "mean_assist_magnitude": float(env.assist_sum[0] / steps),
            "gate_active_fraction": float(env.gate_active_steps[0] / steps),
            "near_zero_assist_fraction": float(env.near_zero_steps[0] / steps),
        }
    last_episode["final_position"] = env.position[0].astype(float).tolist()
    return last_episode


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    survival = np.asarray([episode["survival_time"] for episode in episodes])
    return {
        "episodes": len(episodes),
        "mean_survival_time": float(np.mean(survival)),
        "median_survival_time": float(np.median(survival)),
        "crash_rate": float(np.mean([episode["crashed"] for episode in episodes])),
        "rotor_strike_rate": float(np.mean([episode["reason"] == "rotor_strike" for episode in episodes])),
        "mean_maximum_tilt_degrees": float(np.mean([episode["max_tilt_degrees"] for episode in episodes])),
        "mean_pitch_roll_angular_rate": float(np.mean([episode["mean_pitch_roll_rate"] for episode in episodes])),
        "mean_assist_magnitude": float(np.mean([episode["mean_assist_magnitude"] for episode in episodes])),
        "intervention_gate_active_fraction": float(np.mean([episode["gate_active_fraction"] for episode in episodes])),
        "near_zero_assistance_fraction": float(np.mean([episode["near_zero_assist_fraction"] for episode in episodes])),
    }


def no_input_sink_test(actor: Actor, device: torch.device, seconds: float = 5.0) -> dict[str, float]:
    config = EnvironmentConfig(episode_seconds=seconds + 1.0, domain_randomization=0.0)
    env = VectorizedQwasEnv(1, 7, config)
    snapshot = env.snapshot()
    snapshot["position"][:] = (0.0, 8.0, 0.0)
    snapshot["velocity"][:] = 0.0
    snapshot["orientation"][:] = (0.0, 0.0, 0.0, 1.0)
    snapshot["angular_velocity"][:] = 0.0
    snapshot["player_thrust"][:] = 0.0
    snapshot["applied_thrust"][:] = 0.0
    snapshot["physics"][:] = DEFAULT_PHYSICS
    env.load_snapshot(snapshot)
    vertical_velocities = []
    steps = int(seconds * 60)
    for _ in range(steps):
        env.set_controls(np.zeros((1, 4), dtype=np.float32), 1.0 / 60.0)
        action = _actor_action(actor, env.raw_observation(), device)
        env.step(action, mode="trained", auto_reset=False, resample_controls=False)
        vertical_velocities.append(float(env.velocity[0, 1]))
    tail = vertical_velocities[len(vertical_velocities) // 2:]
    return {
        "mean_vertical_sink_rate_mps": float(-np.mean(tail)),
        "final_vertical_velocity_mps": float(vertical_velocities[-1]),
        "altitude_loss_m": float(8.0 - env.position[0, 1]),
    }


def player_agency_test(actor: Actor, device: torch.device) -> dict[str, Any]:
    config = EnvironmentConfig(episode_seconds=5.0, domain_randomization=0.0)
    base = VectorizedQwasEnv(1, 11, config)
    snapshot = base.snapshot()
    snapshot["position"][:] = (0.0, 2.5, 0.0)
    snapshot["velocity"][:] = 0.0
    snapshot["orientation"][:] = (0.0, 0.0, 0.0, 1.0)
    snapshot["angular_velocity"][:] = 0.0
    snapshot["player_thrust"][:] = 0.0
    snapshot["applied_thrust"][:] = 0.0
    snapshot["physics"][:] = DEFAULT_PHYSICS

    steps = 240
    dt = np.full(steps, 1.0 / 60.0, dtype=np.float32)
    buttons = np.zeros((steps, 4), dtype=np.float32)
    buttons[0:24] = 1.0
    buttons[24:62, 2:4] = 1.0  # rear motors create deliberate forward pitch
    buttons[62:150] = 1.0
    buttons[150:180, 0:2] = 1.0

    outcomes = {}
    for mode in ("no_assist", "trained"):
        scenario = {"snapshot": snapshot, "dt": dt, "buttons": buttons}
        episode = run_scenario(actor, mode, scenario, config, device)
        outcomes[mode] = {
            "forward_displacement_m": float(-episode["final_position"][2]),
            "horizontal_displacement_m": float(np.linalg.norm(np.asarray(episode["final_position"])[[0, 2]])),
            "maximum_tilt_degrees": episode["max_tilt_degrees"],
            "survival_time": episode["survival_time"],
        }
    trained = outcomes["trained"]
    reference = outcomes["no_assist"]
    minimum_displacement = max(0.20, 0.25 * reference["horizontal_displacement_m"])
    passed = trained["maximum_tilt_degrees"] >= 12.0 and trained["horizontal_displacement_m"] >= minimum_displacement
    return {
        "pass": bool(passed),
        "criterion": "Easy Mode reaches >=12 degrees tilt and >=max(0.2 m, 25% of no-assist displacement)",
        **outcomes,
    }


def evaluate_actor(actor: Actor, seeds: Iterable[int], device: torch.device,
                   duration: float = 10.0, include_all_modes: bool = True) -> dict[str, Any]:
    config = EnvironmentConfig(episode_seconds=duration)
    seed_list = list(seeds)
    scenarios = [make_scenario(seed, duration, config) for seed in seed_list]
    modes = ("no_assist", "zero", "trained") if include_all_modes else ("trained",)
    result: dict[str, Any] = {"seeds": seed_list, "duration_seconds": duration, "environment": asdict(config)}
    for mode in modes:
        episodes = [run_scenario(actor, mode, scenario, config, device) for scenario in scenarios]
        result[mode] = summarize_episodes(episodes)
    result["no_input_sink"] = no_input_sink_test(actor, device)
    result["player_agency"] = player_agency_test(actor, device)
    return result
