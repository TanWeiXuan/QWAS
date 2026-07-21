#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from stability_assist.environment import EnvironmentConfig, VectorizedQwasEnv, quaternion_from_body_angles
from stability_assist.model import ActorCritic
from stability_assist.spec import DEFAULT_PHYSICS, OBSERVATION_SCALE, OBSERVATION_SIZE, normalize_observation
from stability_assist.weights import numpy_forward, read_model


def write_inference_fixture(checkpoint_path: Path, model_path: Path, output_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor_critic = ActorCritic()
    actor_critic.load_state_dict(checkpoint["model_state"])
    actor_critic.eval()
    rng = np.random.default_rng(99173)
    raw = np.vstack((
        np.zeros((1, OBSERVATION_SIZE), dtype=np.float32),
        (4.0 * OBSERVATION_SCALE)[None, :],
        (-4.0 * OBSERVATION_SCALE)[None, :],
        rng.uniform(-4.0, 4.0, size=(29, OBSERVATION_SIZE)).astype(np.float32) * OBSERVATION_SCALE,
    ))
    with torch.no_grad():
        expected_torch = actor_critic.actor(torch.from_numpy(normalize_observation(raw))).numpy()
    expected_export = numpy_forward(read_model(model_path), raw)
    maximum_error = float(np.max(np.abs(expected_torch - expected_export)))
    if maximum_error > 1e-6:
        raise RuntimeError(f"checkpoint/export parity failed: {maximum_error}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(raw.shape[0])]
    for observation, action in zip(raw, expected_torch):
        lines.append(" ".join(f"{float(value):.9g}" for value in np.concatenate((observation, action))))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Inference fixture: {output_path} ({raw.shape[0]} cases, max export error {maximum_error:.3g})", flush=True)


def write_physics_fixture(output_path: Path) -> None:
    steps = 180
    config = EnvironmentConfig(episode_seconds=20.0, domain_randomization=0.0)
    env = VectorizedQwasEnv(1, 17, config)
    pitch = np.asarray([np.deg2rad(12.0)], dtype=np.float32)
    yaw = np.asarray([np.deg2rad(-18.0)], dtype=np.float32)
    roll = np.asarray([np.deg2rad(9.0)], dtype=np.float32)
    initial = {
        "position": np.asarray([[0.4, 6.0, -0.7]], dtype=np.float32),
        "velocity": np.asarray([[0.2, -0.1, -0.3]], dtype=np.float32),
        "orientation": quaternion_from_body_angles(pitch, yaw, roll),
        "angular_velocity": np.asarray([[0.3, -0.2, 0.15]], dtype=np.float32),
        "player_thrust": np.asarray([[0.4, 0.7, 0.2, 0.6]], dtype=np.float32),
        "applied_thrust": np.asarray([[0.4, 0.7, 0.2, 0.6]], dtype=np.float32),
        "physics": DEFAULT_PHYSICS[None, :].copy(),
    }
    env.load_snapshot(initial)
    rng = np.random.default_rng(1871)
    dt = rng.uniform(1.0 / 72.0, 1.0 / 48.0, steps).astype(np.float32)
    buttons = np.zeros((steps, 4), dtype=np.float32)
    buttons[0:28, (0, 1)] = 1.0
    buttons[28:74, (2, 3)] = 1.0
    buttons[74:110, :] = 1.0
    buttons[110:145, (0, 3)] = 1.0
    buttons[145:170, (1, 2)] = 1.0
    for step in range(steps):
        env.set_controls(buttons[step][None, :], dt[step])
        env.step(np.zeros((1, 4), dtype=np.float32), mode="no_assist",
                 auto_reset=False, resample_controls=False)
    expected = np.concatenate((env.position[0], env.velocity[0], env.orientation[0],
                               env.angular_velocity[0], env.player_thrust[0]))
    values = [str(steps)]
    values.append(" ".join(f"{float(value):.9g}" for value in np.concatenate((
        initial["position"][0], initial["velocity"][0], initial["orientation"][0],
        initial["angular_velocity"][0], initial["player_thrust"][0], DEFAULT_PHYSICS,
    ))))
    values.extend(" ".join([f"{float(dt_value):.9g}", *(str(int(value)) for value in row)])
                  for dt_value, row in zip(dt, buttons))
    values.append(" ".join(f"{float(value):.9g}" for value in expected))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(values) + "\n", encoding="utf-8")
    print(f"Physics fixture: {output_path} ({steps} steps)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inference-output", type=Path, default=Path("tests/data/stability_assist_parity.txt"))
    parser.add_argument("--physics-output", type=Path, default=Path("tests/data/physics_parity.txt"))
    args = parser.parse_args()
    write_inference_fixture(args.checkpoint, args.model, args.inference_output)
    write_physics_fixture(args.physics_output)


if __name__ == "__main__":
    main()
