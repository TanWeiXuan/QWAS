from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .spec import (
    DEFAULT_PHYSICS,
    EASY_MODE_MAX_DIFFERENTIAL_THRUST_RATIO,
    EASY_MODE_SINK_THRUST_RATIO,
    EASY_MODE_STRENGTH,
    OBSERVATION_SCALE,
    OBSERVATION_SIZE,
    PITCH_MODE,
    ROLL_MODE,
    intervention_gate,
    normalize_observation,
)

MASS, GRAVITY, MAX_THRUST, RAMP_UP, RAMP_DOWN, ARM_LENGTH, I_PITCH, I_YAW, I_ROLL, LIN_DRAG, ANG_DRAG, K_YAW = range(12)


@dataclass
class EnvironmentConfig:
    episode_seconds: float = 10.0
    dt_min: float = 1.0 / 75.0
    dt_max: float = 1.0 / 45.0
    input_hold_min: float = 0.10
    input_hold_max: float = 0.80
    domain_randomization: float = 0.20
    default_physics_probability: float = 0.35
    alive_reward: float = 1.0
    crash_penalty: float = 8.0
    dangerous_tilt_degrees: float = 35.0
    tilt_penalty: float = 4.0
    pitch_roll_rate_penalty: float = 0.15
    assist_magnitude_penalty: float = 0.010
    assist_change_penalty: float = 0.012
    unnecessary_intervention_penalty: float = 0.020


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
    # Compose world yaw with body pitch/roll; all runtime integration below is
    # exactly q <- normalize(q + 0.5 * (q * omega) * dt), as in Drone::Update.
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


class VectorizedQwasEnv:
    """NumPy-vectorized mirror of the C++ QWAS rigid-body update."""

    def __init__(self, count: int, seed: int, config: EnvironmentConfig | None = None) -> None:
        self.count = count
        self.config = config or EnvironmentConfig()
        self.rng = np.random.default_rng(seed)
        shape3 = (count, 3)
        self.position = np.zeros(shape3, dtype=np.float32)
        self.velocity = np.zeros(shape3, dtype=np.float32)
        self.orientation = np.zeros((count, 4), dtype=np.float32)
        self.angular_velocity = np.zeros(shape3, dtype=np.float32)
        self.player_thrust = np.zeros((count, 4), dtype=np.float32)
        self.applied_thrust = np.zeros((count, 4), dtype=np.float32)
        self.buttons = np.zeros((count, 4), dtype=np.float32)
        self.control_time = np.zeros(count, dtype=np.float32)
        self.dt = np.full(count, 1.0 / 60.0, dtype=np.float32)
        self.physics = np.tile(DEFAULT_PHYSICS, (count, 1)).astype(np.float32)
        self.elapsed = np.zeros(count, dtype=np.float32)
        self.previous_assist = np.zeros((count, 4), dtype=np.float32)
        self.max_tilt = np.zeros(count, dtype=np.float32)
        self.rate_sum = np.zeros(count, dtype=np.float32)
        self.assist_sum = np.zeros(count, dtype=np.float32)
        self.gate_active_steps = np.zeros(count, dtype=np.int32)
        self.near_zero_steps = np.zeros(count, dtype=np.int32)
        self.episode_steps = np.zeros(count, dtype=np.int32)
        self.episode_return = np.zeros(count, dtype=np.float32)
        self.reset()

    def reset(self, indices: np.ndarray | None = None) -> None:
        if indices is None:
            indices = np.arange(self.count)
        indices = np.asarray(indices, dtype=np.int64)
        n = indices.size
        if n == 0:
            return

        randomized = self.rng.uniform(1.0 - self.config.domain_randomization,
                                      1.0 + self.config.domain_randomization, size=(n, 12)).astype(np.float32)
        physics = DEFAULT_PHYSICS * randomized
        use_default = self.rng.random(n) < self.config.default_physics_probability
        physics[use_default] = DEFAULT_PHYSICS
        self.physics[indices] = physics

        difficult = self.rng.random(n) < 0.20
        angle_limit = np.where(difficult, np.deg2rad(55.0), np.deg2rad(22.0))
        pitch = self.rng.uniform(-1.0, 1.0, n) * angle_limit
        roll = self.rng.uniform(-1.0, 1.0, n) * angle_limit
        yaw = self.rng.uniform(-np.pi, np.pi, n)
        self.orientation[indices] = quaternion_from_body_angles(pitch, yaw, roll)
        self.position[indices] = 0.0
        self.position[indices, 0] = self.rng.uniform(-1.0, 1.0, n)
        self.position[indices, 1] = self.rng.uniform(3.0, 7.0, n)
        self.velocity[indices] = self.rng.uniform(-0.6, 0.6, size=(n, 3))
        angular_limit = np.where(difficult, 4.0, 1.8)[:, None]
        self.angular_velocity[indices] = self.rng.uniform(-1.0, 1.0, size=(n, 3)) * angular_limit
        maximum = physics[:, MAX_THRUST, None]
        self.player_thrust[indices] = self.rng.uniform(0.0, 0.45, size=(n, 4)) * maximum
        self.applied_thrust[indices] = self.player_thrust[indices]
        self.elapsed[indices] = 0.0
        self.previous_assist[indices] = 0.0
        self.max_tilt[indices] = 0.0
        self.rate_sum[indices] = 0.0
        self.assist_sum[indices] = 0.0
        self.gate_active_steps[indices] = 0
        self.near_zero_steps[indices] = 0
        self.episode_steps[indices] = 0
        self.episode_return[indices] = 0.0
        self.control_time[indices] = 0.0
        self._sample_new_controls(indices)
        self.dt[indices] = self.rng.uniform(self.config.dt_min, self.config.dt_max, n)

    def snapshot(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name).copy() for name in (
            "position", "velocity", "orientation", "angular_velocity",
            "player_thrust", "applied_thrust", "physics",
        )}

    def load_snapshot(self, snapshot: dict[str, np.ndarray]) -> None:
        for name, value in snapshot.items():
            getattr(self, name)[:] = value
        self.elapsed[:] = 0.0
        self.previous_assist[:] = 0.0
        self.max_tilt[:] = 0.0
        self.rate_sum[:] = 0.0
        self.assist_sum[:] = 0.0
        self.gate_active_steps[:] = 0
        self.near_zero_steps[:] = 0
        self.episode_steps[:] = 0
        self.episode_return[:] = 0.0

    def _sample_new_controls(self, indices: np.ndarray) -> None:
        probabilities = np.asarray([0.25, 0.20, 0.25, 0.20, 0.10])
        counts = self.rng.choice(5, size=indices.size, p=probabilities)
        self.buttons[indices] = 0.0
        for row, active_count in zip(indices, counts):
            if active_count:
                active = self.rng.choice(4, size=int(active_count), replace=False)
                self.buttons[row, active] = 1.0
        self.control_time[indices] = self.rng.uniform(
            self.config.input_hold_min, self.config.input_hold_max, indices.size)

    def set_controls(self, buttons: np.ndarray, dt: np.ndarray | float) -> None:
        self.buttons[:] = np.asarray(buttons, dtype=np.float32).reshape(self.count, 4)
        self.dt[:] = np.broadcast_to(np.asarray(dt, dtype=np.float32), (self.count,))

    def _gate_and_up(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        body_up = body_up_from_quaternion(self.orientation)
        tilt = np.arccos(np.clip(body_up[:, 1], -1.0, 1.0)).astype(np.float32)
        pitch_roll_rate = np.linalg.norm(self.angular_velocity[:, (0, 2)], axis=1).astype(np.float32)
        gate = intervention_gate(tilt, pitch_roll_rate)
        return body_up, tilt, pitch_roll_rate, gate

    def raw_observation(self) -> np.ndarray:
        body_up, _, _, gate = self._gate_and_up()
        raw = np.empty((self.count, OBSERVATION_SIZE), dtype=np.float32)
        raw[:, 0:3] = body_up
        raw[:, 3:6] = self.angular_velocity
        raw[:, 6:10] = self.player_thrust / np.maximum(self.physics[:, MAX_THRUST, None], 1e-6)
        raw[:, 10:14] = self.buttons
        raw[:, 14] = self.dt
        raw[:, 15] = gate
        raw[:, 16:28] = self.physics
        return raw

    def observe(self) -> np.ndarray:
        return normalize_observation(self.raw_observation())

    def _apply_thrust(self, raw_action: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        _, tilt, pitch_roll_rate, gate = self._gate_and_up()
        maximum = self.physics[:, MAX_THRUST]
        if mode == "no_assist":
            self.applied_thrust[:] = self.player_thrust
            correction = np.zeros_like(self.player_thrust)
            return correction, tilt, pitch_roll_rate, gate

        player_total = self.player_thrust.sum(axis=1)
        sink_target = EASY_MODE_SINK_THRUST_RATIO * self.physics[:, MASS] * self.physics[:, GRAVITY]
        collective_top_up = np.maximum(0.0, sink_target - player_total) * 0.25
        collective_top_up = np.minimum(
            collective_top_up,
            np.maximum(0.0, maximum[:, None] - self.player_thrust).min(axis=1),
        )
        base = self.player_thrust + collective_top_up[:, None]

        action = np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0)
        if mode == "zero":
            action.fill(0.0)
        pitch_component = (action * PITCH_MODE).sum(axis=1) * 0.25
        roll_component = (action * ROLL_MODE).sum(axis=1) * 0.25
        projected = pitch_component[:, None] * PITCH_MODE + roll_component[:, None] * ROLL_MODE
        correction = projected * (EASY_MODE_MAX_DIFFERENTIAL_THRUST_RATIO * EASY_MODE_STRENGTH * maximum * gate)[:, None]

        feasibility = np.ones(self.count, dtype=np.float32)
        for motor in range(4):
            positive = correction[:, motor] > 0.0
            negative = correction[:, motor] < 0.0
            feasibility[positive] = np.minimum(
                feasibility[positive],
                (maximum[positive] - base[positive, motor]) / correction[positive, motor],
            )
            feasibility[negative] = np.minimum(
                feasibility[negative],
                base[negative, motor] / -correction[negative, motor],
            )
        feasibility = np.clip(feasibility, 0.0, 1.0)
        correction *= feasibility[:, None]
        self.applied_thrust[:] = np.clip(base + correction, 0.0, maximum[:, None])
        return correction, tilt, pitch_roll_rate, gate

    def step(self, raw_action: np.ndarray, mode: str = "trained", *, auto_reset: bool = True,
             resample_controls: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        dt = self.dt
        maximum = self.physics[:, MAX_THRUST, None]
        self.player_thrust[:] = np.where(
            self.buttons > 0.5,
            np.minimum(self.player_thrust + self.physics[:, RAMP_UP, None] * dt[:, None], maximum),
            np.maximum(self.player_thrust - self.physics[:, RAMP_DOWN, None] * dt[:, None], 0.0),
        )
        correction, tilt_before, pitch_roll_rate, gate = self._apply_thrust(raw_action, mode)
        action_intention = np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0)
        intention_pitch = (action_intention * PITCH_MODE).sum(axis=1) * 0.25
        intention_roll = (action_intention * ROLL_MODE).sum(axis=1) * 0.25
        intention = intention_pitch[:, None] * PITCH_MODE + intention_roll[:, None] * ROLL_MODE
        if mode != "trained":
            intention.fill(0.0)

        # These equations deliberately match Drone::Update line-for-line:
        # torque, semi-implicit angular integration + drag, q*(omega), then
        # force/velocity + drag, followed by position integration.
        arm = self.physics[:, ARM_LENGTH]
        thrust = self.applied_thrust
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

        body_up = body_up_from_quaternion(self.orientation)
        acceleration = body_up * (thrust.sum(axis=1) / self.physics[:, MASS])[:, None]
        acceleration[:, 1] -= self.physics[:, GRAVITY]
        self.velocity += acceleration * dt[:, None]
        self.velocity *= (1.0 - self.physics[:, LIN_DRAG] * dt)[:, None]
        self.position += self.velocity * dt[:, None]
        self.elapsed += dt

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
        timed_out = self.elapsed >= self.config.episode_seconds
        done = rotor_strike | body_impact | too_high | out_of_bounds | non_finite | timed_out
        crashed = rotor_strike | body_impact | too_high | out_of_bounds | non_finite

        tilt_excess = np.maximum(0.0, tilt_before - np.deg2rad(self.config.dangerous_tilt_degrees))
        assist_normalized = np.mean(np.abs(correction) / maximum, axis=1)
        assist_change = np.mean(np.abs(correction - self.previous_assist) / maximum, axis=1)
        safe_unnecessary = np.mean(np.abs(intention), axis=1) * (gate < 0.10)
        reward_rate = (
            self.config.alive_reward
            - self.config.tilt_penalty * tilt_excess * tilt_excess
            - self.config.pitch_roll_rate_penalty * pitch_roll_rate * pitch_roll_rate
            - self.config.assist_magnitude_penalty * assist_normalized
            - self.config.assist_change_penalty * assist_change
            - self.config.unnecessary_intervention_penalty * safe_unnecessary
        )
        reward = reward_rate * dt - self.config.crash_penalty * crashed.astype(np.float32)
        self.episode_return += reward
        self.previous_assist[:] = correction

        self.max_tilt = np.maximum(self.max_tilt, tilt_before)
        self.rate_sum += pitch_roll_rate
        self.assist_sum += assist_normalized
        self.gate_active_steps += (gate > 0.01)
        self.near_zero_steps += (assist_normalized < 0.01)
        self.episode_steps += 1

        reasons = np.full(self.count, "timeout", dtype=object)
        reasons[out_of_bounds] = "out_of_bounds"
        reasons[too_high] = "too_high"
        reasons[body_impact] = "body_impact"
        reasons[rotor_strike] = "rotor_strike"
        reasons[non_finite] = "non_finite"
        episodes = []
        for index in np.flatnonzero(done):
            steps = max(int(self.episode_steps[index]), 1)
            episodes.append({
                "index": int(index),
                "survival_time": float(self.elapsed[index]),
                "episode_return": float(self.episode_return[index]),
                "crashed": bool(crashed[index]),
                "reason": str(reasons[index]),
                "max_tilt_degrees": float(np.rad2deg(self.max_tilt[index])),
                "mean_pitch_roll_rate": float(self.rate_sum[index] / steps),
                "mean_assist_magnitude": float(self.assist_sum[index] / steps),
                "gate_active_fraction": float(self.gate_active_steps[index] / steps),
                "near_zero_assist_fraction": float(self.near_zero_steps[index] / steps),
            })

        if auto_reset and np.any(done):
            self.reset(np.flatnonzero(done))
        if resample_controls:
            self.control_time -= dt
            expired = np.flatnonzero(self.control_time <= 0.0)
            if expired.size:
                self._sample_new_controls(expired)
            self.dt[:] = self.rng.uniform(self.config.dt_min, self.config.dt_max, self.count)
        return self.observe(), reward.astype(np.float32), done, {"episodes": episodes}


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
