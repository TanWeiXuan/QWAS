from __future__ import annotations

import re
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.stability_assist.environment import (
    EnvironmentConfig,
    VectorizedQwasEnv,
    compute_release_pd,
    decay_player_differential,
)
from tools.stability_assist.model import Actor
from tools.stability_assist.spec import (
    ACTION_SIZE,
    DEFAULT_PHYSICS,
    EASY_MODE_RELEASE_DIFFERENTIAL_TIME_CONSTANT,
    OBSERVATION_NAMES,
    OBSERVATION_SCALE,
    OBSERVATION_SIZE,
    PITCH_MODE,
    ROLL_MODE,
    parameter_count,
    release_blend,
)
from tools.stability_assist.weights import HEADER, read_model


class StabilityAssistTests(unittest.TestCase):
    def test_observation_and_actor_specification(self) -> None:
        self.assertEqual(OBSERVATION_SIZE, 32)
        self.assertEqual(ACTION_SIZE, 2)
        self.assertEqual(parameter_count(), 2178)
        self.assertEqual(Actor().layers[-2].out_features, 2)
        self.assertEqual(len(OBSERVATION_NAMES), len(OBSERVATION_SCALE))
        self.assertEqual(OBSERVATION_NAMES[28:], (
            "release_blend", "time_since_player_input",
            "previous_pitch_residual", "previous_roll_residual",
        ))

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
            "ASSIST_RELEASE_BLEND", "ASSIST_TIME_SINCE_PLAYER_INPUT",
            "ASSIST_PREVIOUS_PITCH_RESIDUAL", "ASSIST_PREVIOUS_ROLL_RESIDUAL",
        )
        parsed = {name: int(value) for name, value in re.findall(r"(ASSIST_[A-Z_]+)\s*=\s*(\d+)", header)}
        self.assertEqual([parsed[name] for name in cpp_names], list(range(OBSERVATION_SIZE)))

    def test_two_axis_mapping_has_no_collective_or_yaw(self) -> None:
        rng = np.random.default_rng(4)
        actions = rng.uniform(-1.0, 1.0, (100, 2))
        correction = actions[:, 0, None] * PITCH_MODE + actions[:, 1, None] * ROLL_MODE
        yaw_mode = np.asarray([1.0, -1.0, -1.0, 1.0])
        np.testing.assert_allclose(correction.sum(axis=1), 0.0, atol=1e-7)
        np.testing.assert_allclose((correction * yaw_mode).sum(axis=1), 0.0, atol=1e-7)

    def test_release_blend_is_smooth_and_button_resets_timer(self) -> None:
        values = release_blend(np.asarray([0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24]))
        self.assertEqual(float(values[0]), 0.0)
        self.assertEqual(float(values[1]), 0.0)
        self.assertEqual(float(values[-1]), 1.0)
        self.assertTrue(np.all(np.diff(values) >= 0.0))
        self.assertTrue(np.all((values[2:5] > 0.0) & (values[2:5] < 1.0)))

        env = VectorizedQwasEnv(1, 20, EnvironmentConfig(domain_randomization=0.0))
        env.set_controls(np.zeros((1, 4), dtype=np.float32), 0.10)
        env.raw_observation("zero")
        self.assertGreater(float(env.release_blend[0]), 0.0)
        env.step(np.zeros((1, 2), dtype=np.float32), mode="zero", auto_reset=False, resample_controls=False)
        env.set_controls(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), 0.01)
        env.raw_observation("zero")
        self.assertEqual(float(env.time_since_player_input[0]), 0.0)
        self.assertEqual(float(env.release_blend[0]), 0.0)

    def test_release_differential_decay_preserves_collective(self) -> None:
        initial = np.asarray([[0.2, 1.4, 0.6, 1.0]], dtype=np.float32)
        dt = np.asarray([0.05], dtype=np.float32)
        decayed = decay_player_differential(initial.copy(), dt, np.ones(1, dtype=np.float32))
        self.assertAlmostEqual(float(decayed.mean()), float(initial.mean()), places=6)
        initial_span = float(np.ptp(initial))
        expected = initial_span * np.exp(-float(dt[0]) / EASY_MODE_RELEASE_DIFFERENTIAL_TIME_CONSTANT)
        self.assertAlmostEqual(float(np.ptp(decayed)), expected, places=5)
        self.assertLess(float(np.ptp(decayed)), initial_span)

    def test_easy_mode_disabled_preserves_player_thrust_path(self) -> None:
        env = VectorizedQwasEnv(1, 21, EnvironmentConfig(domain_randomization=0.0))
        env.player_thrust[:] = (0.7, 1.1, 0.3, 0.9)
        env.set_controls(np.zeros((1, 4), dtype=np.float32), 0.02)
        expected = np.maximum(env.player_thrust.copy() - DEFAULT_PHYSICS[4] * 0.02, 0.0)
        env.raw_observation("no_assist")
        np.testing.assert_allclose(env.player_thrust, expected, atol=1e-7)
        env.step(np.zeros((1, 2), dtype=np.float32), mode="no_assist",
                 auto_reset=False, resample_controls=False)
        np.testing.assert_allclose(env.applied_thrust, expected, atol=1e-7)
        self.assertEqual(float(env.release_blend[0]), 0.0)

    def test_no_input_top_up_is_below_hover(self) -> None:
        env = VectorizedQwasEnv(1, 2, EnvironmentConfig(domain_randomization=0.0))
        env.orientation[:] = (0.0, 0.0, 0.0, 1.0)
        env.player_thrust[:] = 0.0
        env.physics[:] = DEFAULT_PHYSICS
        env.set_controls(np.zeros((1, 4), dtype=np.float32), 1.0 / 60.0)
        env.step(np.zeros((1, 2), dtype=np.float32), mode="zero", auto_reset=False, resample_controls=False)
        total = float(env.applied_thrust.sum())
        hover = float(DEFAULT_PHYSICS[0] * DEFAULT_PHYSICS[1])
        self.assertAlmostEqual(total, 0.95 * hover, places=5)
        self.assertLess(total, hover)

    def test_common_feasibility_respects_motor_limits(self) -> None:
        env = VectorizedQwasEnv(1, 3, EnvironmentConfig(domain_randomization=0.0))
        env.orientation[:] = (0.0, 0.0, 0.0, 1.0)
        env.player_thrust[:] = (0.5, 0.0, 0.0, 0.0)
        env.physics[:] = DEFAULT_PHYSICS
        env.physics[:, 0] = 5.0
        env.physics[:, 1] = 20.0
        env.physics[:, 2] = 0.5
        env.set_controls(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), 1.0 / 60.0)
        env.step(np.asarray([[1.0, -1.0]], dtype=np.float32), mode="trained",
                 auto_reset=False, resample_controls=False)
        self.assertTrue(np.all(env.applied_thrust >= 0.0))
        self.assertTrue(np.all(env.applied_thrust <= 0.5))

    def test_release_pd_signs_and_level_output(self) -> None:
        angular = np.zeros((1, 3), dtype=np.float32)
        positive_pitch = compute_release_pd(np.asarray([[0.0, 0.98, 0.2]], dtype=np.float32), angular)
        negative_pitch = compute_release_pd(np.asarray([[0.0, 0.98, -0.2]], dtype=np.float32), angular)
        positive_roll = compute_release_pd(np.asarray([[0.2, 0.98, 0.0]], dtype=np.float32), angular)
        negative_roll = compute_release_pd(np.asarray([[-0.2, 0.98, 0.0]], dtype=np.float32), angular)
        self.assertLess(float(positive_pitch[0, 0]), 0.0)
        self.assertGreater(float(negative_pitch[0, 0]), 0.0)
        self.assertGreater(float(positive_roll[0, 1]), 0.0)
        self.assertLess(float(negative_roll[0, 1]), 0.0)
        np.testing.assert_allclose(
            compute_release_pd(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32), angular),
            0.0, atol=1e-7)

    def test_portable_model_validation_and_old_version_rejection(self) -> None:
        model_path = Path("models/stability_assist.qwasmlp")
        if not model_path.exists():
            self.skipTest("portable model has not been generated")
        try:
            read_model(model_path)
        except ValueError:
            self.skipTest("portable v2 model has not been generated")
        data = bytearray(model_path.read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_version = root / "old.qwasmlp"
            bad = bytearray(data[:-264]); struct.pack_into("<I", bad, 8, 1); old_version.write_bytes(bad)
            with self.assertRaisesRegex(ValueError, "version"):
                read_model(old_version)
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
            bad = bytearray(data)
            struct.pack_into("<f", bad, HEADER.size + OBSERVATION_SIZE * 4, float("nan"))
            non_finite.write_bytes(bad)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                read_model(non_finite)


if __name__ == "__main__":
    unittest.main()
