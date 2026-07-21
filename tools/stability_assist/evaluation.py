from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

import numpy as np
import torch

from .environment import (
    EnvironmentConfig,
    VectorizedQwasEnv,
    body_up_from_quaternion,
    generate_held_button_sequence,
    quaternion_from_body_angles,
)
from .model import Actor
from .spec import ACTION_SIZE, DEFAULT_PHYSICS, normalize_observation


def _actor_action(actor: Actor | None, raw_observation: np.ndarray,
                  device: torch.device) -> np.ndarray:
    if actor is None:
        return np.zeros((raw_observation.shape[0], ACTION_SIZE), dtype=np.float32)
    observation = torch.from_numpy(normalize_observation(raw_observation)).to(device)
    with torch.no_grad():
        return actor(observation).cpu().numpy().astype(np.float32)


def _configured_snapshot(seed: int, config: EnvironmentConfig, *, tilt_degrees: float,
                         tilt_azimuth: float, pitch_roll_rate: float,
                         rate_azimuth: float, altitude: float,
                         player_thrust: np.ndarray | None = None) -> dict[str, np.ndarray]:
    env = VectorizedQwasEnv(1, seed, config)
    snapshot = env.snapshot()
    pitch = np.asarray([np.deg2rad(tilt_degrees) * np.cos(tilt_azimuth)], dtype=np.float32)
    roll = np.asarray([np.deg2rad(tilt_degrees) * np.sin(tilt_azimuth)], dtype=np.float32)
    snapshot["position"][:] = (0.0, altitude, 0.0)
    snapshot["velocity"][:] = 0.0
    snapshot["orientation"][:] = quaternion_from_body_angles(
        pitch, np.asarray([0.0], dtype=np.float32), roll)
    snapshot["angular_velocity"][:] = 0.0
    snapshot["angular_velocity"][:, 0] = pitch_roll_rate * np.cos(rate_azimuth)
    snapshot["angular_velocity"][:, 2] = pitch_roll_rate * np.sin(rate_azimuth)
    snapshot["player_thrust"][:] = 0.0 if player_thrust is None else player_thrust
    snapshot["applied_thrust"][:] = snapshot["player_thrust"]
    snapshot["physics"][:] = DEFAULT_PHYSICS
    snapshot["time_since_player_input"][:] = 0.0
    snapshot["release_blend"][:] = 0.0
    snapshot["previous_residual"][:] = 0.0
    body_up = body_up_from_quaternion(snapshot["orientation"])
    snapshot["previous_level_error"][:] = body_up[:, 0] ** 2 + body_up[:, 2] ** 2
    snapshot["previous_rate_error"][:] = (
        snapshot["angular_velocity"][:, 0] ** 2 + snapshot["angular_velocity"][:, 2] ** 2)
    snapshot["settled_time"][:] = 0.0
    return snapshot


def make_random_scenario(seed: int, duration: float, config: EnvironmentConfig) -> dict[str, Any]:
    initial_env = VectorizedQwasEnv(1, seed, config)
    rng = np.random.default_rng(seed + 100_003)
    max_steps = int(np.ceil(duration / config.dt_min)) + 2
    dt = rng.uniform(config.dt_min, config.dt_max, max_steps).astype(np.float32)
    buttons = generate_held_button_sequence(
        seed + 200_003, max_steps, dt, config.input_hold_min, config.input_hold_max)
    return {"snapshot": initial_env.snapshot(), "dt": dt, "buttons": buttons}


def make_release_scenario(seed: int, recovery_seconds: float,
                          config: EnvironmentConfig) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 300_007)
    hard = bool(rng.random() < 0.20)
    # The held input creates the final release disturbance. Most cases begin
    # moderately displaced so precision settling, rather than only saturation,
    # dominates the suite; one fifth begin in a harder state.
    tilt = float(rng.uniform(20.0, 40.0) if hard else rng.uniform(0.0, 15.0))
    rate = float(rng.uniform(1.0, 2.2) if hard else rng.uniform(0.0, 0.8))
    maximum = float(DEFAULT_PHYSICS[2])
    player_thrust = rng.uniform(0.05, 0.32, 4).astype(np.float32) * maximum
    snapshot = _configured_snapshot(
        seed, config, tilt_degrees=tilt, tilt_azimuth=float(rng.uniform(-np.pi, np.pi)),
        pitch_roll_rate=rate, rate_azimuth=float(rng.uniform(-np.pi, np.pi)),
        altitude=float(rng.uniform(4.5, 8.0)), player_thrust=player_thrust)
    snapshot["velocity"][:] = rng.uniform(-0.25, 0.25, 3)
    pre_release = float(rng.uniform(0.20, 0.55 if hard else 0.70))
    total = pre_release + recovery_seconds
    max_steps = int(np.ceil(total / config.dt_min)) + 4
    dt = rng.uniform(config.dt_min, config.dt_max, max_steps).astype(np.float32)
    buttons = np.zeros((max_steps, 4), dtype=np.float32)
    active_count = int(rng.integers(1, 3))
    held = rng.choice(4, size=active_count, replace=False)
    elapsed = 0.0
    for step, frame_dt in enumerate(dt):
        if elapsed < pre_release:
            buttons[step, held] = 1.0
        elapsed += float(frame_dt)
    return {
        "snapshot": snapshot, "dt": dt, "buttons": buttons,
        "pre_release_seconds": pre_release, "recovery_seconds": recovery_seconds,
        "initial_tilt_degrees": tilt, "initial_pitch_roll_rate": rate,
    }


def _episode_from_env(env: VectorizedQwasEnv, *, crashed: bool = False,
                      reason: str = "duration") -> dict[str, Any]:
    steps = max(int(env.episode_steps[0]), 1)
    body_up = body_up_from_quaternion(env.orientation)[0]
    tilt = float(np.rad2deg(np.arccos(np.clip(body_up[1], -1.0, 1.0))))
    rate = float(np.linalg.norm(env.angular_velocity[0, (0, 2)]))
    return {
        "survival_time": float(env.elapsed[0]), "crashed": crashed, "reason": reason,
        "max_tilt_degrees": float(np.rad2deg(env.max_tilt[0])),
        "mean_pitch_roll_rate": float(env.rate_sum[0] / steps),
        "mean_assist_magnitude": float(env.assist_sum[0] / steps),
        "mean_residual_magnitude": float(env.residual_sum[0] / steps),
        "gate_active_fraction": float(env.gate_active_steps[0] / steps),
        "near_zero_assist_fraction": float(env.near_zero_steps[0] / steps),
        "pd_saturated_fraction": float(env.pd_saturated_steps[0] / steps),
        "final_tilt_degrees": tilt, "final_pitch_roll_rate": rate,
        "final_position": env.position[0].astype(float).tolist(),
    }


def run_random_scenario(actor: Actor | None, mode: str, scenario: dict[str, Any],
                        config: EnvironmentConfig, device: torch.device) -> dict[str, Any]:
    env = VectorizedQwasEnv(1, 0, config)
    env.load_snapshot(scenario["snapshot"])
    episode: dict[str, Any] | None = None
    for buttons, dt in zip(scenario["buttons"], scenario["dt"]):
        if float(env.elapsed[0]) >= config.episode_seconds:
            break
        env.set_controls(buttons[None, :], float(dt))
        action = _actor_action(actor if mode == "trained" else None,
                               env.raw_observation(mode), device)
        _, _, done, info = env.step(
            action, mode=mode, auto_reset=False, resample_controls=False)
        if done[0]:
            episode = info["episodes"][0]
            break
    if episode is None:
        episode = _episode_from_env(env)
    episode["final_position"] = env.position[0].astype(float).tolist()
    return episode


def summarize_random_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    survival = np.asarray([episode["survival_time"] for episode in episodes])
    return {
        "episodes": len(episodes),
        "mean_survival_time": float(np.mean(survival)),
        "median_survival_time": float(np.median(survival)),
        "crash_rate": float(np.mean([episode["crashed"] for episode in episodes])),
        "rotor_strike_rate": float(np.mean(
            [episode["reason"] == "rotor_strike" for episode in episodes])),
        "mean_maximum_tilt_degrees": float(np.mean(
            [episode["max_tilt_degrees"] for episode in episodes])),
        "mean_pitch_roll_angular_rate": float(np.mean(
            [episode["mean_pitch_roll_rate"] for episode in episodes])),
        "mean_assist_magnitude": float(np.mean(
            [episode["mean_assist_magnitude"] for episode in episodes])),
        "mean_residual_action_magnitude": float(np.mean(
            [episode.get("mean_residual_magnitude", 0.0) for episode in episodes])),
        "intervention_gate_active_fraction": float(np.mean(
            [episode["gate_active_fraction"] for episode in episodes])),
        "near_zero_assistance_fraction": float(np.mean(
            [episode["near_zero_assist_fraction"] for episode in episodes])),
    }


def run_release_scenario(actor: Actor | None, mode: str, scenario: dict[str, Any],
                         config: EnvironmentConfig, device: torch.device) -> dict[str, Any]:
    env = VectorizedQwasEnv(1, 0, config)
    env.load_snapshot(scenario["snapshot"])
    release_time: float | None = None
    release_tilt = 0.0
    settled_time: float | None = None
    crashed = False
    max_post_tilt = 0.0
    assist: list[float] = []
    residual: list[float] = []
    pd_saturated: list[float] = []
    final_tilt = 180.0
    final_rate = 100.0
    previous_active = True
    for buttons, dt in zip(scenario["buttons"], scenario["dt"]):
        active = bool(np.any(buttons > 0.5))
        if previous_active and not active:
            release_time = float(env.elapsed[0])
            body_up = body_up_from_quaternion(env.orientation)[0]
            release_tilt = float(np.rad2deg(np.arccos(np.clip(body_up[1], -1.0, 1.0))))
        previous_active = active
        env.set_controls(buttons[None, :], float(dt))
        action = _actor_action(actor if mode == "trained" else None,
                               env.raw_observation(mode), device)
        _, _, done, info = env.step(
            action, mode=mode, auto_reset=False, resample_controls=False)
        telemetry = info["telemetry"]
        if release_time is not None:
            final_tilt = float(np.rad2deg(telemetry["tilt"][0]))
            final_rate = float(telemetry["rate"][0])
            max_post_tilt = max(max_post_tilt, final_tilt)
            assist.append(float(telemetry["assist_magnitude"][0]))
            residual.append(float(telemetry["residual_magnitude"][0]))
            pd_saturated.append(float(telemetry["pd_saturated"][0]))
            if settled_time is None and bool(telemetry["settled"][0]):
                settled_time = float(env.elapsed[0]) - release_time
            if float(env.elapsed[0]) - release_time >= scenario["recovery_seconds"]:
                break
        if done[0]:
            crashed = bool(info["episodes"][0]["crashed"])
            break
    recovery_window = float(scenario["recovery_seconds"])
    success = settled_time is not None and not crashed
    capped_settle = min(settled_time, recovery_window) if success else recovery_window
    return {
        "success": success, "settling_time_seconds": float(capped_settle),
        "release_tilt_degrees": release_tilt,
        "final_tilt_degrees": final_tilt,
        "final_level_error": float(np.sin(np.deg2rad(final_tilt)) ** 2),
        "final_pitch_roll_rate": final_rate,
        "maximum_post_release_tilt_degrees": max_post_tilt,
        "attitude_overshoot_degrees": max(0.0, max_post_tilt - release_tilt),
        "crashed_before_settling": bool(crashed and not success),
        "mean_assist_magnitude": float(np.mean(assist)) if assist else 0.0,
        "mean_residual_action_magnitude": float(np.mean(residual)) if residual else 0.0,
        "pd_saturated_any": bool(any(pd_saturated)),
        "pd_saturated_step_fraction": float(np.mean(pd_saturated)) if pd_saturated else 0.0,
    }


def summarize_release_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    settling = np.asarray([case["settling_time_seconds"] for case in cases])
    return {
        "cases": len(cases),
        "recovery_success_rate": float(np.mean([case["success"] for case in cases])),
        "mean_settling_time_seconds": float(np.mean(settling)),
        "median_settling_time_seconds": float(np.median(settling)),
        "p90_settling_time_seconds": float(np.percentile(settling, 90)),
        "mean_final_tilt_degrees": float(np.mean([case["final_tilt_degrees"] for case in cases])),
        "mean_final_level_error": float(np.mean([case["final_level_error"] for case in cases])),
        "mean_final_pitch_roll_rate": float(np.mean(
            [case["final_pitch_roll_rate"] for case in cases])),
        "mean_maximum_post_release_tilt_degrees": float(np.mean(
            [case["maximum_post_release_tilt_degrees"] for case in cases])),
        "mean_attitude_overshoot_degrees": float(np.mean(
            [case["attitude_overshoot_degrees"] for case in cases])),
        "crash_rate_before_settling": float(np.mean(
            [case["crashed_before_settling"] for case in cases])),
        "mean_assist_magnitude": float(np.mean([case["mean_assist_magnitude"] for case in cases])),
        "mean_residual_action_magnitude": float(np.mean(
            [case["mean_residual_action_magnitude"] for case in cases])),
        "pd_saturated_case_fraction": float(np.mean([case["pd_saturated_any"] for case in cases])),
        "pd_saturated_step_fraction": float(np.mean(
            [case["pd_saturated_step_fraction"] for case in cases])),
    }


def _run_no_input_case(actor: Actor, device: torch.device, seed: int,
                       seconds: float) -> dict[str, Any]:
    config = EnvironmentConfig(episode_seconds=seconds + 1.0, domain_randomization=0.0)
    rng = np.random.default_rng(seed)
    player = rng.uniform(0.05, 0.28, 4).astype(np.float32) * DEFAULT_PHYSICS[2]
    snapshot = _configured_snapshot(
        seed, config, tilt_degrees=float(rng.uniform(7.0, 28.0)),
        tilt_azimuth=float(rng.uniform(-np.pi, np.pi)),
        pitch_roll_rate=float(rng.uniform(0.2, 2.2)),
        rate_azimuth=float(rng.uniform(-np.pi, np.pi)), altitude=8.0,
        player_thrust=player)
    env = VectorizedQwasEnv(1, seed + 1, config)
    env.load_snapshot(snapshot)
    initial_altitude = float(env.position[0, 1])
    settled_at: float | None = None
    tail: list[tuple[float, float, float]] = []
    crashed = False
    for _ in range(int(seconds * 60)):
        env.set_controls(np.zeros((1, 4), dtype=np.float32), 1.0 / 60.0)
        action = _actor_action(actor, env.raw_observation("trained"), device)
        _, _, done, info = env.step(
            action, mode="trained", auto_reset=False, resample_controls=False)
        telemetry = info["telemetry"]
        if settled_at is None and bool(telemetry["settled"][0]):
            settled_at = float(env.elapsed[0])
        if settled_at is not None:
            tail.append((float(telemetry["vertical_velocity"][0]),
                         float(np.rad2deg(telemetry["tilt"][0])),
                         float(telemetry["rate"][0])))
        if done[0]:
            crashed = bool(info["episodes"][0]["crashed"])
            break
    if not tail:
        body_up = body_up_from_quaternion(env.orientation)[0]
        tail = [(float(env.velocity[0, 1]),
                 float(np.rad2deg(np.arccos(np.clip(body_up[1], -1.0, 1.0)))),
                 float(np.linalg.norm(env.angular_velocity[0, (0, 2)])))]
    return {
        "recovered": settled_at is not None and not crashed,
        "time_to_recover_seconds": settled_at if settled_at is not None else seconds,
        "mean_settled_vertical_velocity_mps": float(np.mean([item[0] for item in tail])),
        "mean_settled_tilt_degrees": float(np.mean([item[1] for item in tail])),
        "mean_settled_pitch_roll_rate": float(np.mean([item[2] for item in tail])),
        "altitude_increased": bool(float(env.position[0, 1]) > initial_altitude + 0.02),
        "altitude_change_m": float(env.position[0, 1] - initial_altitude),
        "crashed": crashed,
    }


def no_input_sink_test(actor: Actor, device: torch.device, seconds: float = 5.0,
                       seeds: Iterable[int] = range(7200, 7208)) -> dict[str, Any]:
    cases = [_run_no_input_case(actor, device, seed, seconds) for seed in seeds]
    vertical = np.asarray([case["mean_settled_vertical_velocity_mps"] for case in cases])
    return {
        "cases": len(cases),
        "recovery_success_rate": float(np.mean([case["recovered"] for case in cases])),
        "mean_time_to_recover_seconds": float(np.mean(
            [case["time_to_recover_seconds"] for case in cases])),
        "mean_vertical_velocity_after_settling_mps": float(np.mean(vertical)),
        "mean_vertical_sink_rate_mps": float(-np.mean(vertical)),
        "mean_tilt_after_settling_degrees": float(np.mean(
            [case["mean_settled_tilt_degrees"] for case in cases])),
        "mean_pitch_roll_rate_after_settling": float(np.mean(
            [case["mean_settled_pitch_roll_rate"] for case in cases])),
        "altitude_increasing_fraction": float(np.mean(
            [case["altitude_increased"] for case in cases])),
        "crash_rate": float(np.mean([case["crashed"] for case in cases])),
        "pass": bool(np.mean(vertical) < 0.0 and not any(case["altitude_increased"] for case in cases)),
    }


def _run_agency_script(actor: Actor | None, mode: str, snapshot: dict[str, np.ndarray],
                       buttons: np.ndarray, device: torch.device,
                       active_steps: int) -> dict[str, Any]:
    config = EnvironmentConfig(episode_seconds=len(buttons) / 60.0 + 0.5, domain_randomization=0.0)
    env = VectorizedQwasEnv(1, 19, config)
    env.load_snapshot(snapshot)
    max_active_tilt = 0.0
    active_position = np.zeros(3, dtype=np.float32)
    settled_after_release = False
    crashed = False
    final_tilt = 180.0
    for step, frame_buttons in enumerate(buttons):
        env.set_controls(frame_buttons[None, :], 1.0 / 60.0)
        action = _actor_action(actor if mode == "trained" else None,
                               env.raw_observation(mode), device)
        _, _, done, info = env.step(
            action, mode=mode, auto_reset=False, resample_controls=False)
        tilt = float(np.rad2deg(info["telemetry"]["tilt"][0]))
        final_tilt = tilt
        if step < active_steps:
            max_active_tilt = max(max_active_tilt, tilt)
            active_position = env.position[0].copy()
        else:
            settled_after_release |= bool(info["telemetry"]["settled"][0])
        if done[0]:
            crashed = bool(info["episodes"][0]["crashed"])
            break
    return {
        "maximum_active_tilt_degrees": max_active_tilt,
        "active_horizontal_displacement_m": float(np.linalg.norm(active_position[[0, 2]])),
        "settled_after_release": settled_after_release,
        "final_tilt_degrees": final_tilt, "crashed": crashed,
    }


def player_agency_test(actor: Actor, device: torch.device) -> dict[str, Any]:
    config = EnvironmentConfig(episode_seconds=6.0, domain_randomization=0.0)
    snapshot = _configured_snapshot(
        11, config, tilt_degrees=0.0, tilt_azimuth=0.0,
        pitch_roll_rate=0.0, rate_azimuth=0.0, altitude=10.0)
    manoeuvre_steps = 300
    active_steps = 39
    buttons = np.zeros((manoeuvre_steps, 4), dtype=np.float32)
    buttons[0:9] = 1.0
    buttons[9:39, 2:4] = 1.0
    outcomes = {
        mode: _run_agency_script(actor, mode, snapshot, buttons, device, active_steps)
        for mode in ("no_assist", "trained")
    }

    aggressive_buttons = np.zeros((300, 4), dtype=np.float32)
    aggressive_buttons[:, 0] = 1.0
    aggressive_snapshot = {name: value.copy() for name, value in snapshot.items()}
    aggressive_snapshot["position"][:, 1] = 2.5
    aggressive = _run_agency_script(
        actor, "trained", aggressive_snapshot, aggressive_buttons, device, len(aggressive_buttons))
    trained = outcomes["trained"]
    reference = outcomes["no_assist"]
    displacement_floor = max(0.10, 0.20 * reference["active_horizontal_displacement_m"])
    pass_active = (trained["maximum_active_tilt_degrees"] >= 12.0 and
                   trained["active_horizontal_displacement_m"] >= displacement_floor)
    passed = pass_active and trained["settled_after_release"] and aggressive["crashed"]
    return {
        "pass": bool(passed),
        "active_manoeuvre_pass": bool(pass_active),
        "release_recovery_pass": bool(trained["settled_after_release"]),
        "aggressive_crash_pass": bool(aggressive["crashed"]),
        "criterion": ("active tilt >=12 deg, horizontal movement >=max(0.10 m, 20% of no-assist), "
                      "settles after release, and sustained single-motor input can crash"),
        "no_assist": reference, "trained": trained, "aggressive_trained": aggressive,
    }


def evaluate_actor(actor: Actor, seeds: Iterable[int], device: torch.device,
                   duration: float = 8.0, include_all_modes: bool = True) -> dict[str, Any]:
    config = EnvironmentConfig(episode_seconds=duration)
    seed_list = list(seeds)
    modes = ("no_assist", "pd_only", "zero", "trained") if include_all_modes else ("trained",)
    random_scenarios = [make_random_scenario(seed, duration, config) for seed in seed_list]
    recovery_seconds = min(4.5, max(3.0, duration - 1.0))
    release_scenarios = [
        make_release_scenario(seed + 10_000, recovery_seconds, config) for seed in seed_list]
    random_results: dict[str, Any] = {}
    release_results: dict[str, Any] = {
        "settling_time_note": "Failures are capped at the fixed recovery window.",
        "recovery_window_seconds": recovery_seconds,
    }
    for mode in modes:
        random_episodes = [
            run_random_scenario(actor, mode, scenario, config, device)
            for scenario in random_scenarios]
        random_results[mode] = summarize_random_episodes(random_episodes)
        release_cases = [
            run_release_scenario(actor, mode, scenario, config, device)
            for scenario in release_scenarios]
        release_results[mode] = summarize_release_cases(release_cases)

    if include_all_modes:
        pd = release_results["pd_only"]
        trained = release_results["trained"]
        release_results["trained_residual_improvement_over_pd_only"] = {
            "recovery_success_percentage_points": 100.0 * (
                trained["recovery_success_rate"] - pd["recovery_success_rate"]),
            "median_settling_time_reduction_seconds": (
                pd["median_settling_time_seconds"] - trained["median_settling_time_seconds"]),
            "final_tilt_reduction_degrees": (
                pd["mean_final_tilt_degrees"] - trained["mean_final_tilt_degrees"]),
            "final_rate_reduction": (
                pd["mean_final_pitch_roll_rate"] - trained["mean_final_pitch_roll_rate"]),
        }
    result: dict[str, Any] = {
        "seeds": seed_list, "duration_seconds": duration,
        "environment": asdict(config), "random_player": random_results,
        "release_recovery": release_results,
        "no_input_sink": no_input_sink_test(actor, device),
        "player_agency": player_agency_test(actor, device),
    }
    # Preserve the convenient v1 top-level random-player keys for CLI/report consumers.
    result.update(random_results)
    return result
