from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .spec import (
    DEFAULT_PHYSICS,
    EASY_MODE_ACTIVE_DIFFERENTIAL_THRUST_RATIO,
    EASY_MODE_RELEASE_DIFFERENTIAL_THRUST_RATIO,
    EASY_MODE_RELEASE_DIFFERENTIAL_TIME_CONSTANT,
    EASY_MODE_RELEASE_PITCH_KD,
    EASY_MODE_RELEASE_PITCH_KP,
    EASY_MODE_RELEASE_RESIDUAL_SCALE,
    EASY_MODE_RELEASE_ROLL_KD,
    EASY_MODE_RELEASE_ROLL_KP,
    EASY_MODE_SINK_THRUST_RATIO,
    OBSERVATION_SIZE,
    PITCH_MODE,
    ROLL_MODE,
    danger_gate,
    normalize_observation,
    release_blend,
    release_recovery_gate,
)

MASS, GRAVITY, MAX_THRUST, RAMP_UP, RAMP_DOWN, ARM_LENGTH, I_PITCH, I_YAW, I_ROLL, LIN_DRAG, ANG_DRAG, K_YAW = range(12)
GENERAL_SCENARIO = 0
RELEASE_SCENARIO = 1


@dataclass
class EnvironmentConfig:
    episode_seconds: float = 8.0
    dt_min: float = 1.0 / 75.0
    dt_max: float = 1.0 / 30.0
    input_hold_min: float = 0.10
    input_hold_max: float = 0.80
    release_scenario_probability: float = 0.60
    release_perturb_min: float = 0.20
    release_perturb_max: float = 1.20
    release_recovery_min: float = 2.0
    release_recovery_max: float = 4.0
    domain_randomization: float = 0.20
    default_physics_probability: float = 0.35
    settle_tilt_degrees: float = 5.0
    settle_rate: float = 0.25
    settle_hold_seconds: float = 0.25
    alive_reward: float = 1.0
    crash_penalty: float = 8.0
    assist_magnitude_penalty: float = 0.010
    action_change_penalty: float = 0.010
    active_dangerous_tilt_degrees: float = 35.0
    active_tilt_penalty: float = 4.0
    active_rate_penalty: float = 0.12
    active_unnecessary_assist_penalty: float = 0.020
    release_level_penalty: float = 5.0
    release_rate_penalty: float = 0.35
    recovery_progress_reward: float = 2.0
    recovery_rate_progress_weight: float = 0.15
    settled_bonus: float = 0.35


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack((
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ), axis=-1).astype(np.float32)


def _axis_quaternion(axis: int, angle: np.ndarray) -> np.ndarray:
    result = np.zeros((angle.size, 4), dtype=np.float32)
    result[:, axis] = np.sin(angle * 0.5)
    result[:, 3] = np.cos(angle * 0.5)
    return result


def quaternion_from_body_angles(pitch_x: np.ndarray, yaw_y: np.ndarray, roll_z: np.ndarray) -> np.ndarray:
    q_yaw = _axis_quaternion(1, yaw_y)
    q_pitch = _axis_quaternion(0, pitch_x)
    q_roll = _axis_quaternion(2, roll_z)
    return _quat_multiply(_quat_multiply(q_yaw, q_pitch), q_roll)


def body_up_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    return np.stack((
        2.0 * (x * y - w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + w * x),
    ), axis=-1).astype(np.float32)


def compute_release_pd(body_up: np.ndarray, angular_velocity: np.ndarray) -> np.ndarray:
    return np.stack((
        -EASY_MODE_RELEASE_PITCH_KP * body_up[:, 2] - EASY_MODE_RELEASE_PITCH_KD * angular_velocity[:, 0],
        +EASY_MODE_RELEASE_ROLL_KP * body_up[:, 0] - EASY_MODE_RELEASE_ROLL_KD * angular_velocity[:, 2],
    ), axis=1).astype(np.float32)


def decay_player_differential(player_thrust: np.ndarray, dt: np.ndarray,
                              blend: np.ndarray) -> np.ndarray:
    mean = player_thrust.mean(axis=1, keepdims=True)
    exponential = np.exp(-dt / EASY_MODE_RELEASE_DIFFERENTIAL_TIME_CONSTANT)
    factor = 1.0 - np.clip(blend, 0.0, 1.0) * (1.0 - exponential)
    return np.maximum(0.0, mean + (player_thrust - mean) * factor[:, None]).astype(np.float32)


class VectorizedQwasEnv:
    """Vectorized QWAS physics plus the exact v2 active/release controller."""

    def __init__(self, count: int, seed: int, config: EnvironmentConfig | None = None) -> None:
        self.count = count
        self.config = config or EnvironmentConfig()
        self.rng = np.random.default_rng(seed)
        self.curriculum_progress = 1.0
        self.position = np.zeros((count, 3), dtype=np.float32)
        self.velocity = np.zeros((count, 3), dtype=np.float32)
        self.orientation = np.zeros((count, 4), dtype=np.float32)
        self.angular_velocity = np.zeros((count, 3), dtype=np.float32)
        self.player_thrust = np.zeros((count, 4), dtype=np.float32)
        self.applied_thrust = np.zeros((count, 4), dtype=np.float32)
        self.buttons = np.zeros((count, 4), dtype=np.float32)
        self.control_time = np.zeros(count, dtype=np.float32)
        self.release_window = np.zeros(count, dtype=np.float32)
        self.scenario_type = np.zeros(count, dtype=np.int8)
        self.scenario_phase = np.zeros(count, dtype=np.int8)
        self.dt = np.full(count, 1.0 / 60.0, dtype=np.float32)
        self.physics = np.tile(DEFAULT_PHYSICS, (count, 1)).astype(np.float32)
        self.elapsed = np.zeros(count, dtype=np.float32)
        self.time_since_player_input = np.zeros(count, dtype=np.float32)
        self.release_blend = np.zeros(count, dtype=np.float32)
        self.previous_residual = np.zeros((count, 2), dtype=np.float32)
        self.previous_correction = np.zeros((count, 4), dtype=np.float32)
        self.previous_level_error = np.zeros(count, dtype=np.float32)
        self.previous_rate_error = np.zeros(count, dtype=np.float32)
        self.settled_time = np.zeros(count, dtype=np.float32)
        self.max_tilt = np.zeros(count, dtype=np.float32)
        self.rate_sum = np.zeros(count, dtype=np.float32)
        self.assist_sum = np.zeros(count, dtype=np.float32)
        self.residual_sum = np.zeros(count, dtype=np.float32)
        self.gate_active_steps = np.zeros(count, dtype=np.int32)
        self.near_zero_steps = np.zeros(count, dtype=np.int32)
        self.pd_saturated_steps = np.zeros(count, dtype=np.int32)
        self.episode_steps = np.zeros(count, dtype=np.int32)
        self.episode_return = np.zeros(count, dtype=np.float32)
        self._prepared = False
        self._prepared_mode = ""
        self._cached: dict[str, np.ndarray] = {}
        self.reset()

    def set_curriculum(self, progress: float) -> None:
        self.curriculum_progress = float(np.clip(progress, 0.0, 1.0))

    def reset(self, indices: np.ndarray | None = None) -> None:
        if indices is None:
            indices = np.arange(self.count)
        indices = np.asarray(indices, dtype=np.int64)
        n = indices.size
        if n == 0:
            return
        p = self.curriculum_progress
        domain = 0.05 + (self.config.domain_randomization - 0.05) * min(1.0, p * 1.5)
        randomized = self.rng.uniform(1.0 - domain, 1.0 + domain, size=(n, 12)).astype(np.float32)
        physics = DEFAULT_PHYSICS * randomized
        default_probability = 0.65 + (self.config.default_physics_probability - 0.65) * min(1.0, p * 1.5)
        physics[self.rng.random(n) < default_probability] = DEFAULT_PHYSICS
        self.physics[indices] = physics

        is_release = self.rng.random(n) < self.config.release_scenario_probability
        self.scenario_type[indices] = np.where(is_release, RELEASE_SCENARIO, GENERAL_SCENARIO)
        self.scenario_phase[indices] = 0
        difficult = self.rng.random(n) < (0.05 + 0.15 * p)
        moderate_limit = np.deg2rad(25.0 + 20.0 * p)
        hard_limit = np.deg2rad(45.0 + 25.0 * p)
        angle_limit = np.where(difficult, hard_limit, moderate_limit)
        angle_floor = np.where(is_release, np.deg2rad(5.0), 0.0)
        tilt = self.rng.uniform(0.0, 1.0, n) * (angle_limit - angle_floor) + angle_floor
        azimuth = self.rng.uniform(-np.pi, np.pi, n)
        pitch = tilt * np.cos(azimuth)
        roll = tilt * np.sin(azimuth)
        yaw = self.rng.uniform(-np.pi, np.pi, n)
        self.orientation[indices] = quaternion_from_body_angles(pitch, yaw, roll)
        self.position[indices] = 0.0
        self.position[indices, 0] = self.rng.uniform(-1.0, 1.0, n)
        self.position[indices, 1] = self.rng.uniform(3.0, 8.0, n)
        self.velocity[indices] = self.rng.uniform(-0.6, 0.6, size=(n, 3))
        moderate_rate = 1.5 + 1.5 * p
        hard_rate = 3.0 + 3.0 * p
        rate_limit = np.where(difficult, hard_rate, moderate_rate)
        rate_angle = self.rng.uniform(-np.pi, np.pi, n)
        rate_magnitude = self.rng.uniform(0.0, 1.0, n) * rate_limit
        self.angular_velocity[indices, 0] = rate_magnitude * np.cos(rate_angle)
        self.angular_velocity[indices, 1] = self.rng.uniform(-1.0, 1.0, n)
        self.angular_velocity[indices, 2] = rate_magnitude * np.sin(rate_angle)
        self.player_thrust[indices] = self.rng.uniform(0.0, 0.40, size=(n, 4)) * physics[:, MAX_THRUST, None]
        self.applied_thrust[indices] = self.player_thrust[indices]
        self.buttons[indices] = 0.0
        release_indices = indices[is_release]
        for row in release_indices:
            active_count = int(self.rng.integers(1, 4))
            self.buttons[row, self.rng.choice(4, size=active_count, replace=False)] = 1.0
        self.control_time[release_indices] = self.rng.uniform(
            self.config.release_perturb_min, self.config.release_perturb_max, release_indices.size)
        self.release_window[release_indices] = self.rng.uniform(
            self.config.release_recovery_min, self.config.release_recovery_max, release_indices.size)
        general_indices = indices[~is_release]
        self._sample_new_controls(general_indices)
        self.dt[indices] = self.rng.uniform(self.config.dt_min, self.config.dt_max, n)
        for name in ("elapsed", "time_since_player_input", "release_blend", "settled_time",
                     "max_tilt", "rate_sum", "assist_sum", "residual_sum", "episode_return"):
            getattr(self, name)[indices] = 0.0
        for name in ("previous_residual", "previous_correction"):
            getattr(self, name)[indices] = 0.0
        for name in ("gate_active_steps", "near_zero_steps", "pd_saturated_steps", "episode_steps"):
            getattr(self, name)[indices] = 0
        body_up = body_up_from_quaternion(self.orientation[indices])
        self.previous_level_error[indices] = body_up[:, 0] ** 2 + body_up[:, 2] ** 2
        self.previous_rate_error[indices] = (
            self.angular_velocity[indices, 0] ** 2 + self.angular_velocity[indices, 2] ** 2)
        self._prepared = False

    def snapshot(self) -> dict[str, np.ndarray]:
        names = (
            "position", "velocity", "orientation", "angular_velocity", "player_thrust",
            "applied_thrust", "physics", "time_since_player_input", "release_blend",
            "previous_residual", "previous_level_error", "previous_rate_error", "settled_time",
        )
        return {name: getattr(self, name).copy() for name in names}

    def load_snapshot(self, snapshot: dict[str, np.ndarray]) -> None:
        for name, value in snapshot.items():
            getattr(self, name)[:] = value
        for name in ("elapsed", "max_tilt", "rate_sum", "assist_sum", "residual_sum", "episode_return"):
            getattr(self, name)[:] = 0.0
        for name in ("gate_active_steps", "near_zero_steps", "pd_saturated_steps", "episode_steps"):
            getattr(self, name)[:] = 0
        self.previous_correction[:] = 0.0
        self._prepared = False

    def _sample_new_controls(self, indices: np.ndarray) -> None:
        if indices.size == 0:
            return
        probabilities = np.asarray([0.25, 0.20, 0.25, 0.20, 0.10])
        counts = self.rng.choice(5, size=indices.size, p=probabilities)
        self.buttons[indices] = 0.0
        for row, active_count in zip(indices, counts):
            if active_count:
                self.buttons[row, self.rng.choice(4, size=int(active_count), replace=False)] = 1.0
        self.control_time[indices] = self.rng.uniform(
            self.config.input_hold_min, self.config.input_hold_max, indices.size)

    def set_controls(self, buttons: np.ndarray, dt: np.ndarray | float) -> None:
        self.buttons[:] = np.asarray(buttons, dtype=np.float32).reshape(self.count, 4)
        self.dt[:] = np.broadcast_to(np.asarray(dt, dtype=np.float32), (self.count,))
        self._prepared = False

    def _gate_and_up(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        body_up = body_up_from_quaternion(self.orientation)
        tilt = np.arccos(np.clip(body_up[:, 1], -1.0, 1.0)).astype(np.float32)
        rate = np.linalg.norm(self.angular_velocity[:, (0, 2)], axis=1).astype(np.float32)
        danger = danger_gate(tilt, rate)
        recovery = release_recovery_gate(tilt, rate)
        effective = np.maximum(danger, self.release_blend * recovery).astype(np.float32)
        return body_up, tilt, rate, danger, recovery, effective

    def _prepare(self, mode: str) -> None:
        if self._prepared:
            if mode != self._prepared_mode:
                raise RuntimeError("cannot change assist mode during a prepared step")
            return
        dt = self.dt
        maximum = self.physics[:, MAX_THRUST, None]
        self.player_thrust[:] = np.where(
            self.buttons > 0.5,
            np.minimum(self.player_thrust + self.physics[:, RAMP_UP, None] * dt[:, None], maximum),
            np.maximum(self.player_thrust - self.physics[:, RAMP_DOWN, None] * dt[:, None], 0.0),
        )
        assisted = mode != "no_assist"
        if assisted:
            any_button = np.any(self.buttons > 0.5, axis=1)
            self.time_since_player_input[:] = np.where(any_button, 0.0, self.time_since_player_input + dt)
            self.release_blend[:] = release_blend(self.time_since_player_input)
            released = ~any_button
            if np.any(released):
                self.player_thrust[released] = decay_player_differential(
                    self.player_thrust[released], dt[released], self.release_blend[released])
        else:
            self.time_since_player_input[:] = 0.0
            self.release_blend[:] = 0.0
            self.previous_residual[:] = 0.0

        body_up, tilt, rate, danger, recovery, effective = self._gate_and_up()
        if assisted:
            player_total = self.player_thrust.sum(axis=1)
            sink_target = EASY_MODE_SINK_THRUST_RATIO * self.physics[:, MASS] * self.physics[:, GRAVITY]
            top_up = np.maximum(0.0, sink_target - player_total) * 0.25
            top_up = np.minimum(top_up, np.maximum(0.0, maximum - self.player_thrust).min(axis=1))
        else:
            top_up = np.zeros(self.count, dtype=np.float32)
        self._cached = {
            "body_up": body_up, "tilt": tilt, "rate": rate, "danger": danger,
            "recovery": recovery, "effective": effective,
            "base": self.player_thrust + top_up[:, None],
        }
        self._prepared = True
        self._prepared_mode = mode

    def _raw_observation_from_state(self) -> np.ndarray:
        body_up, _, _, _, _, effective = self._gate_and_up()
        raw = np.empty((self.count, OBSERVATION_SIZE), dtype=np.float32)
        raw[:, 0:3] = body_up
        raw[:, 3:6] = self.angular_velocity
        raw[:, 6:10] = self.player_thrust / np.maximum(self.physics[:, MAX_THRUST, None], 1e-6)
        raw[:, 10:14] = self.buttons
        raw[:, 14] = self.dt
        raw[:, 15] = effective
        raw[:, 16:28] = self.physics
        raw[:, 28] = self.release_blend
        raw[:, 29] = self.time_since_player_input
        raw[:, 30:32] = self.previous_residual
        return raw

    def raw_observation(self, mode: str = "trained") -> np.ndarray:
        self._prepare(mode)
        return self._raw_observation_from_state()

    def observe(self, mode: str = "trained") -> np.ndarray:
        return normalize_observation(self.raw_observation(mode))

    def _apply_action(self, raw_action: np.ndarray, mode: str) -> dict[str, np.ndarray]:
        base = self._cached["base"]
        maximum = self.physics[:, MAX_THRUST]
        if mode == "no_assist":
            self.applied_thrust[:] = self.player_thrust
            zeros = np.zeros(self.count, dtype=np.float32)
            return {"correction": np.zeros_like(self.player_thrust), "residual_magnitude": zeros,
                    "pd_saturated": np.zeros(self.count, dtype=bool), "pitch_command": zeros,
                    "roll_command": zeros}
        residual = np.clip(np.asarray(raw_action, dtype=np.float32).reshape(self.count, 2), -1.0, 1.0)
        if mode in ("pd_only", "zero"):
            residual.fill(0.0)
        pd = compute_release_pd(self._cached["body_up"], self.angular_velocity)
        danger = self._cached["danger"]
        recovery = self._cached["recovery"]
        blend = self.release_blend
        active = danger[:, None] * residual
        release = recovery[:, None] * (pd + EASY_MODE_RELEASE_RESIDUAL_SCALE * residual)
        command = np.clip((1.0 - blend[:, None]) * active + blend[:, None] * release, -1.0, 1.0)
        authority = ((1.0 - blend) * EASY_MODE_ACTIVE_DIFFERENTIAL_THRUST_RATIO +
                     blend * EASY_MODE_RELEASE_DIFFERENTIAL_THRUST_RATIO) * maximum
        correction = (command[:, 0, None] * PITCH_MODE + command[:, 1, None] * ROLL_MODE) * authority[:, None]
        feasibility = np.ones(self.count, dtype=np.float32)
        for motor in range(4):
            positive = correction[:, motor] > 0.0
            negative = correction[:, motor] < 0.0
            feasibility[positive] = np.minimum(
                feasibility[positive],
                (maximum[positive] - base[positive, motor]) / correction[positive, motor])
            feasibility[negative] = np.minimum(
                feasibility[negative], base[negative, motor] / -correction[negative, motor])
        feasibility = np.clip(feasibility, 0.0, 1.0)
        correction *= feasibility[:, None]
        self.applied_thrust[:] = np.clip(base + correction, 0.0, maximum[:, None])
        self.previous_residual[:] = residual
        return {
            "correction": correction,
            "residual_magnitude": np.mean(np.abs(residual), axis=1),
            "pd_saturated": np.any(np.abs(pd) > 1.0, axis=1),
            "pitch_command": command[:, 0],
            "roll_command": command[:, 1],
        }

    def _advance_scenarios(self, dt: np.ndarray) -> np.ndarray:
        complete = np.zeros(self.count, dtype=bool)
        release_rows = np.flatnonzero(self.scenario_type == RELEASE_SCENARIO)
        for row in release_rows:
            self.control_time[row] -= dt[row]
            if self.scenario_phase[row] == 0 and self.control_time[row] <= 0.0:
                self.scenario_phase[row] = 1
                self.buttons[row] = 0.0
                self.control_time[row] = self.release_window[row]
            elif self.scenario_phase[row] == 1 and self.control_time[row] <= 0.0:
                complete[row] = True
        general_rows = np.flatnonzero(self.scenario_type == GENERAL_SCENARIO)
        self.control_time[general_rows] -= dt[general_rows]
        expired = general_rows[self.control_time[general_rows] <= 0.0]
        self._sample_new_controls(expired)
        return complete

    def step(self, raw_action: np.ndarray, mode: str = "trained", *, auto_reset: bool = True,
             resample_controls: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        self._prepare(mode)
        dt = self.dt.copy()
        telemetry = self._apply_action(raw_action, mode)
        thrust = self.applied_thrust
        arm = self.physics[:, ARM_LENGTH]
        tau_x = arm * (thrust[:, 0] + thrust[:, 1] - thrust[:, 2] - thrust[:, 3])
        tau_z = arm * (-thrust[:, 0] + thrust[:, 1] - thrust[:, 2] + thrust[:, 3])
        tau_y = self.physics[:, K_YAW] * (thrust[:, 0] + thrust[:, 3] - thrust[:, 1] - thrust[:, 2])
        self.angular_velocity[:, 0] += tau_x / self.physics[:, I_PITCH] * dt
        self.angular_velocity[:, 1] += tau_y / self.physics[:, I_YAW] * dt
        self.angular_velocity[:, 2] += tau_z / self.physics[:, I_ROLL] * dt
        self.angular_velocity *= (1.0 - self.physics[:, ANG_DRAG] * dt)[:, None]
        omega = np.column_stack((self.angular_velocity, np.zeros(self.count, dtype=np.float32)))
        derivative = _quat_multiply(self.orientation, omega)
        self.orientation += 0.5 * derivative * dt[:, None]
        self.orientation /= np.maximum(np.linalg.norm(self.orientation, axis=1, keepdims=True), 1e-12)
        body_up_after = body_up_from_quaternion(self.orientation)
        acceleration = body_up_after * (thrust.sum(axis=1) / self.physics[:, MASS])[:, None]
        acceleration[:, 1] -= self.physics[:, GRAVITY]
        self.velocity += acceleration * dt[:, None]
        self.velocity *= (1.0 - self.physics[:, LIN_DRAG] * dt)[:, None]
        self.position += self.velocity * dt[:, None]
        self.elapsed += dt

        tilt_after = np.arccos(np.clip(body_up_after[:, 1], -1.0, 1.0)).astype(np.float32)
        rate_after = np.linalg.norm(self.angular_velocity[:, (0, 2)], axis=1).astype(np.float32)
        level_error = body_up_after[:, 0] ** 2 + body_up_after[:, 2] ** 2
        rate_error = self.angular_velocity[:, 0] ** 2 + self.angular_velocity[:, 2] ** 2
        settled_now = (tilt_after < np.deg2rad(self.config.settle_tilt_degrees)) & (rate_after < self.config.settle_rate)
        self.settled_time[:] = np.where(settled_now, self.settled_time + dt, 0.0)
        settled = self.settled_time >= self.config.settle_hold_seconds

        x, y, z, w = np.moveaxis(self.orientation, -1, 0)
        rotation_y_x = 2.0 * (x * y + z * w)
        rotation_y_z = 2.0 * (y * z - x * w)
        local_x = np.asarray([-1.0, 1.0, -1.0, 1.0])[None, :] * arm[:, None]
        local_z = np.asarray([-1.0, -1.0, 1.0, 1.0])[None, :] * arm[:, None]
        rotor_y = self.position[:, 1, None] + rotation_y_x[:, None] * local_x + rotation_y_z[:, None] * local_z
        rotor_strike = np.min(rotor_y, axis=1) < 0.0
        body_impact = self.position[:, 1] < 0.0
        too_high = self.position[:, 1] > 15.0
        out_of_bounds = (np.abs(self.position[:, 0]) > 20.0) | (np.abs(self.position[:, 2]) > 35.0)
        all_state = np.concatenate((self.position, self.velocity, self.orientation, self.angular_velocity), axis=1)
        non_finite = ~np.all(np.isfinite(all_state), axis=1)
        scenario_complete = self._advance_scenarios(dt) if resample_controls else np.zeros(self.count, dtype=bool)
        timed_out = self.elapsed >= self.config.episode_seconds
        done = rotor_strike | body_impact | too_high | out_of_bounds | non_finite | timed_out | scenario_complete
        crashed = rotor_strike | body_impact | too_high | out_of_bounds | non_finite

        assist_magnitude = np.mean(
            np.abs(telemetry["correction"]) / np.maximum(self.physics[:, MAX_THRUST, None], 1e-6), axis=1)
        action_change = np.mean(np.abs(telemetry["correction"] - self.previous_correction) /
                                np.maximum(self.physics[:, MAX_THRUST, None], 1e-6), axis=1)
        active_weight = 1.0 - self.release_blend
        tilt_excess = np.maximum(0.0, self._cached["tilt"] - np.deg2rad(self.config.active_dangerous_tilt_degrees))
        safe_unnecessary = telemetry["residual_magnitude"] * (self._cached["danger"] < 0.10)
        recovery_progress = ((self.previous_level_error - level_error) +
                             self.config.recovery_rate_progress_weight * (self.previous_rate_error - rate_error))
        reward_rate = (
            self.config.alive_reward
            - self.config.assist_magnitude_penalty * assist_magnitude
            - self.config.action_change_penalty * action_change
            - active_weight * self.config.active_tilt_penalty * tilt_excess * tilt_excess
            - active_weight * self.config.active_rate_penalty * self._cached["rate"] ** 2
            - active_weight * self.config.active_unnecessary_assist_penalty * safe_unnecessary
            - self.release_blend * self.config.release_level_penalty * level_error
            - self.release_blend * self.config.release_rate_penalty * rate_error
            + self.release_blend * self.config.settled_bonus * settled.astype(np.float32)
        )
        reward = reward_rate * dt + self.release_blend * self.config.recovery_progress_reward * recovery_progress
        reward -= self.config.crash_penalty * crashed.astype(np.float32)
        self.episode_return += reward
        self.previous_correction[:] = telemetry["correction"]
        self.previous_level_error[:] = level_error
        self.previous_rate_error[:] = rate_error
        self.max_tilt = np.maximum(self.max_tilt, tilt_after)
        self.rate_sum += rate_after
        self.assist_sum += assist_magnitude
        self.residual_sum += telemetry["residual_magnitude"]
        self.gate_active_steps += (self._cached["effective"] > 0.01)
        self.near_zero_steps += (assist_magnitude < 0.01)
        self.pd_saturated_steps += telemetry["pd_saturated"]
        self.episode_steps += 1

        reasons = np.full(self.count, "duration", dtype=object)
        reasons[out_of_bounds] = "out_of_bounds"
        reasons[too_high] = "too_high"
        reasons[body_impact] = "body_impact"
        reasons[rotor_strike] = "rotor_strike"
        reasons[non_finite] = "non_finite"
        episodes = []
        for index in np.flatnonzero(done):
            steps = max(int(self.episode_steps[index]), 1)
            episodes.append({
                "index": int(index), "survival_time": float(self.elapsed[index]),
                "episode_return": float(self.episode_return[index]), "crashed": bool(crashed[index]),
                "reason": str(reasons[index]), "scenario": "release" if self.scenario_type[index] else "general",
                "max_tilt_degrees": float(np.rad2deg(self.max_tilt[index])),
                "mean_pitch_roll_rate": float(self.rate_sum[index] / steps),
                "mean_assist_magnitude": float(self.assist_sum[index] / steps),
                "mean_residual_magnitude": float(self.residual_sum[index] / steps),
                "gate_active_fraction": float(self.gate_active_steps[index] / steps),
                "near_zero_assist_fraction": float(self.near_zero_steps[index] / steps),
                "pd_saturated_fraction": float(self.pd_saturated_steps[index] / steps),
                "final_tilt_degrees": float(np.rad2deg(tilt_after[index])),
                "final_pitch_roll_rate": float(rate_after[index]),
                "settled": bool(settled[index]),
            })

        self._prepared = False
        if auto_reset and np.any(done):
            self.reset(np.flatnonzero(done))
        if resample_controls:
            self.dt[:] = self.rng.uniform(self.config.dt_min, self.config.dt_max, self.count)
            next_observation = self.observe(mode)
        else:
            next_observation = normalize_observation(self._raw_observation_from_state())
        info = {
            "episodes": episodes,
            "telemetry": {
                "tilt": tilt_after.copy(), "rate": rate_after.copy(),
                "release_blend": self.release_blend.copy(), "assist_magnitude": assist_magnitude.copy(),
                "residual_magnitude": telemetry["residual_magnitude"].copy(),
                "pd_saturated": telemetry["pd_saturated"].copy(), "settled": settled.copy(),
                "vertical_velocity": self.velocity[:, 1].copy(),
            },
        }
        return next_observation, reward.astype(np.float32), done, info


def generate_held_button_sequence(seed: int, steps: int, dt: np.ndarray,
                                  hold_min: float = 0.10, hold_max: float = 0.80) -> np.ndarray:
    rng = np.random.default_rng(seed)
    buttons = np.zeros((steps, 4), dtype=np.float32)
    current = np.zeros(4, dtype=np.float32)
    remaining = 0.0
    probabilities = np.asarray([0.25, 0.20, 0.25, 0.20, 0.10])
    for step in range(steps):
        if remaining <= 0.0:
            current.fill(0.0)
            count = int(rng.choice(5, p=probabilities))
            if count:
                current[rng.choice(4, size=count, replace=False)] = 1.0
            remaining = float(rng.uniform(hold_min, hold_max))
        buttons[step] = current
        remaining -= float(dt[step])
    return buttons
