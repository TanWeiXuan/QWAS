from __future__ import annotations

import numpy as np

OBSERVATION_NAMES = (
    "body_up_world_x", "body_up_world_y", "body_up_world_z",
    "body_angular_velocity_x", "body_angular_velocity_y", "body_angular_velocity_z",
    "player_thrust_front_left_normalized", "player_thrust_front_right_normalized",
    "player_thrust_rear_left_normalized", "player_thrust_rear_right_normalized",
    "button_front_left", "button_front_right", "button_rear_left", "button_rear_right",
    "frame_dt", "intervention_gate",
    "mass", "gravity", "maximum_thrust", "thrust_ramp_up", "thrust_ramp_down",
    "arm_length", "pitch_inertia", "yaw_inertia", "roll_inertia",
    "linear_drag", "angular_drag", "yaw_coefficient",
    "release_blend", "time_since_player_input",
    "previous_pitch_residual", "previous_roll_residual",
)

OBSERVATION_SIZE = len(OBSERVATION_NAMES)
HIDDEN_SIZE = 32
ACTION_SIZE = 2
ARCHITECTURE = (OBSERVATION_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, ACTION_SIZE)

PHYSICS_NAMES = (
    "mass", "gravity", "maximum_thrust", "thrust_ramp_up", "thrust_ramp_down",
    "arm_length", "pitch_inertia", "yaw_inertia", "roll_inertia",
    "linear_drag", "angular_drag", "yaw_coefficient",
)

DEFAULT_PHYSICS = np.asarray(
    [0.5, 9.81, 5.0, 10.0, 10.0, 0.25, 0.4, 0.7, 0.4, 1.0, 0.5, 0.02],
    dtype=np.float32,
)

# Raw values are divided by these scales, then clamped to [-4, 4]. Player
# thrust is already normalized by the current maximum thrust before this step.
OBSERVATION_SCALE = np.asarray(
    [1.0, 1.0, 1.0, 5.0, 5.0, 5.0,
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
     1.0 / 60.0, 1.0,
     *DEFAULT_PHYSICS.tolist(),
     1.0, 1.0, 1.0, 1.0],
    dtype=np.float32,
)

PITCH_MODE = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
ROLL_MODE = np.asarray([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)

EASY_MODE_SINK_THRUST_RATIO = 0.95
EASY_MODE_RELEASE_BLEND_START = 0.04
EASY_MODE_RELEASE_BLEND_FULL = 0.20
EASY_MODE_RELEASE_DIFFERENTIAL_TIME_CONSTANT = 0.14
EASY_MODE_ACTIVE_DIFFERENTIAL_THRUST_RATIO = 0.30
EASY_MODE_RELEASE_DIFFERENTIAL_THRUST_RATIO = 0.50
EASY_MODE_RELEASE_PITCH_KP = 1.80
EASY_MODE_RELEASE_PITCH_KD = 0.70
EASY_MODE_RELEASE_ROLL_KP = 1.80
EASY_MODE_RELEASE_ROLL_KD = 0.70
EASY_MODE_RELEASE_RESIDUAL_SCALE = 0.20
TILT_GATE_START = np.deg2rad(20.0)
TILT_GATE_FULL = np.deg2rad(60.0)
RATE_GATE_START = 1.5
RATE_GATE_FULL = 5.0
RELEASE_TILT_GATE_START = np.deg2rad(1.5)
RELEASE_TILT_GATE_FULL = np.deg2rad(20.0)
RELEASE_RATE_GATE_START = 0.12
RELEASE_RATE_GATE_FULL = 2.0


def normalize_observation(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    return np.clip(values / OBSERVATION_SCALE, -4.0, 4.0).astype(np.float32, copy=False)


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def danger_gate(tilt_radians: np.ndarray, pitch_roll_rate: np.ndarray) -> np.ndarray:
    tilt = _smoothstep(TILT_GATE_START, TILT_GATE_FULL, tilt_radians)
    rate = _smoothstep(RATE_GATE_START, RATE_GATE_FULL, pitch_roll_rate)
    return np.maximum(tilt, rate).astype(np.float32, copy=False)


def release_recovery_gate(tilt_radians: np.ndarray, pitch_roll_rate: np.ndarray) -> np.ndarray:
    tilt = _smoothstep(RELEASE_TILT_GATE_START, RELEASE_TILT_GATE_FULL, tilt_radians)
    rate = _smoothstep(RELEASE_RATE_GATE_START, RELEASE_RATE_GATE_FULL, pitch_roll_rate)
    return np.maximum(tilt, rate).astype(np.float32, copy=False)


def release_blend(time_since_player_input: np.ndarray) -> np.ndarray:
    return _smoothstep(EASY_MODE_RELEASE_BLEND_START, EASY_MODE_RELEASE_BLEND_FULL,
                       time_since_player_input).astype(np.float32, copy=False)


def parameter_count() -> int:
    return OBSERVATION_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * ACTION_SIZE + ACTION_SIZE
