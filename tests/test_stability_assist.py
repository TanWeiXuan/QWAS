from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
import re

import numpy as np

from tools.stability_assist.environment import EnvironmentConfig, VectorizedQwasEnv
from tools.stability_assist.spec import (
    DEFAULT_PHYSICS, OBSERVATION_NAMES, OBSERVATION_SCALE, OBSERVATION_SIZE,
    PITCH_MODE, ROLL_MODE,
)
from tools.stability_assist.weights import HEADER, read_model


class StabilityAssistTests(unittest.TestCase):
    def test_observation_specification(self) -> None:
        self.assertEqual(OBSERVATION_SIZE, 28)
        self.assertEqual(len(OBSERVATION_NAMES), len(OBSERVATION_SCALE))
        self.assertEqual(OBSERVATION_NAMES[16:], tuple((
            "mass", "gravity", "maximum_thrust", "thrust_ramp_up", "thrust_ramp_down",
            "arm_length", "pitch_inertia", "yaw_inertia", "roll_inertia",
            "linear_drag", "angular_drag", "yaw_coefficient",
        )))

    def test_cpp_observation_indices_match(self) -> None:
        header = Path("include/stability_assist.h").read_text(encoding="utf-8")
        cpp_names = (
            "ASSIST_BODY_UP_X", "ASSIST_BODY_UP_Y", "ASSIST_BODY_UP_Z",
            "ASSIST_ANGULAR_VELOCITY_X", "ASSIST_ANGULAR_VELOCITY_Y", "ASSIST_ANGULAR_VELOCITY_Z",
            "ASSIST_PLAYER_THRUST_FRONT_LEFT", "ASSIST_PLAYER_THRUST_FRONT_RIGHT",
            "ASSIST_PLAYER_THRUST_REAR_LEFT", "ASSIST_PLAYER_THRUST_REAR_RIGHT",
            "ASSIST_BUTTON_FRONT_LEFT", "ASSIST_BUTTON_FRONT_RIGHT", "ASSIST_BUTTON_REAR_LEFT", "ASSIST_BUTTON_REAR_RIGHT",
            "ASSIST_FRAME_DT", "ASSIST_INTERVENTION_GATE", "ASSIST_MASS", "ASSIST_GRAVITY",
            "ASSIST_MAXIMUM_THRUST", "ASSIST_THRUST_RAMP_UP", "ASSIST_THRUST_RAMP_DOWN",
            "ASSIST_ARM_LENGTH", "ASSIST_PITCH_INERTIA", "ASSIST_YAW_INERTIA", "ASSIST_ROLL_INERTIA",
            "ASSIST_LINEAR_DRAG", "ASSIST_ANGULAR_DRAG", "ASSIST_YAW_COEFFICIENT",
        )
        parsed = {name: int(value) for name, value in re.findall(r"(ASSIST_[A-Z_]+)\s*=\s*(\d+)", header)}
        self.assertEqual([parsed[name] for name in cpp_names], list(range(OBSERVATION_SIZE)))

    def test_projection_has_no_collective_or_yaw(self) -> None:
        rng = np.random.default_rng(4)
        raw = rng.uniform(-1.0, 1.0, (100, 4))
        pitch = (raw * PITCH_MODE).sum(axis=1) * 0.25
        roll = (raw * ROLL_MODE).sum(axis=1) * 0.25
        correction = pitch[:, None] * PITCH_MODE + roll[:, None] * ROLL_MODE
        yaw_mode = np.asarray([1.0, -1.0, -1.0, 1.0])
        np.testing.assert_allclose(correction.sum(axis=1), 0.0, atol=1e-7)
        np.testing.assert_allclose((correction * yaw_mode).sum(axis=1), 0.0, atol=1e-7)

    def test_no_input_top_up_is_below_hover(self) -> None:
        env = VectorizedQwasEnv(1, 2, EnvironmentConfig(domain_randomization=0.0))
        env.orientation[:] = (0.0, 0.0, 0.0, 1.0)
        env.player_thrust[:] = 0.0
        env.physics[:] = DEFAULT_PHYSICS
        env.set_controls(np.zeros((1, 4), dtype=np.float32), 1.0 / 60.0)
        env.step(np.zeros((1, 4), dtype=np.float32), mode="zero", auto_reset=False, resample_controls=False)
        total = float(env.applied_thrust.sum())
        hover = float(DEFAULT_PHYSICS[0] * DEFAULT_PHYSICS[1])
        self.assertAlmostEqual(total, 0.95 * hover, places=5)
        self.assertLess(total, hover)

    def test_infeasible_top_up_respects_motor_limits(self) -> None:
        env = VectorizedQwasEnv(1, 3, EnvironmentConfig(domain_randomization=0.0))
        env.orientation[:] = (0.0, 0.0, 0.0, 1.0)
        env.player_thrust[:] = (0.5, 0.0, 0.0, 0.0)
        env.physics[:] = DEFAULT_PHYSICS
        env.physics[:, 0] = 5.0
        env.physics[:, 1] = 20.0
        env.physics[:, 2] = 0.5
        env.set_controls(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), 1.0 / 60.0)
        env.step(np.zeros((1, 4), dtype=np.float32), mode="zero", auto_reset=False, resample_controls=False)
        self.assertTrue(np.all(env.applied_thrust >= 0.0))
        self.assertTrue(np.all(env.applied_thrust <= 0.5))

    def test_portable_model_validation(self) -> None:
        model_path = Path("models/stability_assist.qwasmlp")
        if not model_path.exists():
            self.skipTest("portable model has not been generated")
        data = bytearray(model_path.read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid_magic = root / "magic.qwasmlp"
            bad = bytearray(data); bad[0] ^= 0xFF; invalid_magic.write_bytes(bad)
            with self.assertRaisesRegex(ValueError, "magic"):
                read_model(invalid_magic)
            truncated = root / "truncated.qwasmlp"; truncated.write_bytes(data[:-4])
            with self.assertRaisesRegex(ValueError, "length"):
                read_model(truncated)
            dimensions = root / "dimensions.qwasmlp"
            bad = bytearray(data); struct.pack_into("<I", bad, 12, 999); dimensions.write_bytes(bad)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                read_model(dimensions)
            non_finite = root / "nonfinite.qwasmlp"
            bad = bytearray(data); struct.pack_into("<f", bad, HEADER.size + OBSERVATION_SIZE * 4, float("nan")); non_finite.write_bytes(bad)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                read_model(non_finite)


if __name__ == "__main__":
    unittest.main()
