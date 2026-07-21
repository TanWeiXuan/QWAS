# Easy Mode stability assist v2

Easy Mode is an optional pitch/roll stability controller. It is off on every application launch and is never an autopilot: the player still creates lift, yaw, translation, and landing motion with the four motors. The controller has no position, goal, landing-zone, or navigation inputs.

## Why v1 recovered too late

The v1 actor was primarily an emergency anti-flip controller. Its single gate stayed off below about 20 degrees of tilt and 1.5 rad/s pitch/roll rate, training only penalized tilt beyond 35 degrees, zero-input periods were incidental, stale player differential thrust used the ordinary ramp-down, and checkpoint selection emphasized survival. It could delay some crashes but did not define “release means return upright.”

V2 separates active-player and released-player behavior, adds deterministic release PD, and uses the network only as a bounded residual.

## Runtime controller

`Drone::playerThrust` remains the persistent output of the existing button ramp. `Drone::appliedThrust` is recomputed for physics, rendering, animation, and the HUD. Easy Mode first applies the normal player ramp, then the following controller.

### Active and release regimes

While any motor button is held, `timeSincePlayerInput` is zero and release authority is inactive. The learned residual is multiplied by the high-threshold danger gate, allowing ordinary intentional tilt and movement.

With no button held, the timer advances and the release blend uses smoothstep from 0 at 0.04 seconds to 1 at 0.20 seconds. A later button press immediately resets both. Controller state and previous residuals are also reset with the drone/level.

After the ordinary ramp-down, release-only differential decay preserves the four-motor mean while reducing stale torque:

```text
mean = average(playerThrust)
factor = 1 - releaseBlend * (1 - exp(-dt / 0.14 s))
playerThrust[i] = max(0, mean + (playerThrust[i] - mean) * factor)
```

This operation adds no collective thrust and creates no yaw component.

### Collective thrust and gates

The deterministic collective target is unchanged in purpose:

```text
sink target = 0.95 * mass * gravity
top-up per motor = max(0, sink target - sum(playerThrust)) / 4
```

Headroom limits make it feasible. Since the ratio is below 1, a level no-input drone descends instead of hovering.

The active danger gate is the maximum of smoothstep gates over 20–60 degrees tilt and 1.5–5.0 rad/s pitch/roll rate. The release-recovery gate uses 1.5–20 degrees and 0.12–2.0 rad/s. The observed effective gate is:

```text
max(dangerGate, releaseBlend * releaseRecoveryGate)
```

There is no hard minimum around level, avoiding persistent oscillation.

### PD plus learned residual

Release PD targets world-up and zero body pitch/roll rate, using gains selected by a deterministic sweep:

```text
pitchPD = -1.8 * bodyUp.z - 0.7 * angularVelocity.x
rollPD  = +1.8 * bodyUp.x - 0.7 * angularVelocity.z
```

The MLP emits only normalized pitch and roll residuals. Active commands use `dangerGate * residual`. Release commands use `releaseRecoveryGate * (PD + 0.20 * residual)`, and the release blend interpolates between them. Normalized commands are clamped before mapping to:

```text
pitch mode = [ 1,  1, -1, -1]
roll mode  = [-1,  1, -1,  1]
```

The maximum differential authority blends from 0.30 of per-motor maximum thrust while active to 0.50 after release. One common feasibility scale keeps every motor within limits. The motor basis has exactly zero collective and yaw components; individual clipping is only a numerical safeguard.

The final tuned constants are therefore: 0.04–0.20 second release blend, 0.14 second differential time constant, `kp=1.8`, `kd=0.7`, active/release authority 0.30/0.50, and residual scale 0.20.

## Actor and observations

Raw observations use fixed scales stored with the model, then clamp to `[-4, 4]`. There is no running normalization.

| Index | Raw value | Scale |
| ---: | --- | ---: |
| 0–2 | body-up world X, Y, Z | 1 each |
| 3–5 | body angular velocity pitch, yaw, roll | 5 rad/s each |
| 6–9 | normalized player thrust FL, FR, RL, RR | 1 each |
| 10–13 | player buttons FL, FR, RL, RR | 1 each |
| 14 | frame `dt` | 1/60 s |
| 15 | effective intervention gate | 1 |
| 16 | mass | 0.5 kg |
| 17 | gravity | 9.81 m/s² |
| 18 | maximum thrust | 5 N |
| 19 | thrust ramp up | 10 N/s |
| 20 | thrust ramp down | 10 N/s |
| 21 | arm length | 0.25 m |
| 22 | pitch inertia | 0.4 kg·m² |
| 23 | yaw inertia | 0.7 kg·m² |
| 24 | roll inertia | 0.4 kg·m² |
| 25 | linear drag | 1/s |
| 26 | angular drag | 0.5/s |
| 27 | yaw coefficient | 0.02 |
| 28 | release blend | 1 |
| 29 | time since player input | 1 s |
| 30–31 | previous pitch and roll residual | 1 each |

The deployed actor is `32 -> tanh(32) -> tanh(32) -> tanh(2)`, with 2,178 weights and biases. It uses fixed-size C++ arrays, no per-frame allocation, and no inference dependency. The critic and trainable two-dimensional Gaussian standard deviation are training-only.

## Training environment and scenarios

`tools/stability_assist/environment.py` mirrors the C++ ramping, release state, differential decay, gates, top-up, PD/residual blend, feasibility scale, and drone physics. Frame time is randomized from 1/75 to 1/30 second and all 12 physics parameters are randomized up to ±20%.

The mixture is 60% perturb-and-release and 40% general random-player scenarios.

- Perturb-and-release holds one to three deliberately unbalanced motors for 0.2–1.2 seconds, releases for 2–4 seconds, and samples 3–8 m altitude. Most cases cover 5–45 degree/moderate-rate states; a hard fraction reaches about 70 degrees and 6 rad/s.
- General scenarios retain held zero-, one-, two-, three-, and four-button patterns, including balanced and unbalanced input, with 0.1–0.8 second hold times.

The explicit curriculum starts with ±5% physics, 65% default-physics resets, moderate release tilt/rate limits, and a 5% hard fraction. Through the first two thirds it expands to ±20% physics, 35% default resets, moderate limits near 45 degrees/3 rad/s, hard limits near 70 degrees/6 rad/s, and a 20% hard fraction. Moderate precision cases remain throughout.

## Reward and warm start

Shared reward coefficients are `+1.0/s` alive, `-8.0` crash, `-0.010` assist magnitude, and `-0.010` assist change. Non-finite state is a terminal crash.

While active, the controller receives no reward for being perfectly level. It uses a 35-degree soft threshold with `-4.0 * excessTilt²`, `-0.12 * pitchRollRate²`, and `-0.020` unnecessary residual magnitude when the danger gate is below 0.1.

Release-conditioned terms are:

```text
- 5.0 * releaseBlend * (bodyUp.x² + bodyUp.z²)
- 0.35 * releaseBlend * (angularVelocity.x² + angularVelocity.z²)
+ 2.0 * releaseBlend * (level progress + 0.15 * rate progress)
+ 0.35/s * releaseBlend after settling
```

Settling requires tilt below 5 degrees and pitch/roll rate below 0.25 rad/s continuously for 0.25 seconds.

There is no reward for forward progress, horizontal displacement, forward velocity, facing/reaching the goal, landing, altitude gain, or hovering.

The 300-batch warm start targets only a small (maximum 0.20) nonlinear residual based on body-up and pitch/roll rates. It randomizes release blend, elapsed release time, player thrust/buttons, physics, frame time, and previous residuals. It does not duplicate full PD and contains no position or navigation signal.

## PPO and checkpoint selection

The full reproducible run used seed `20260721`, 64 environments, 128 rollout steps, 4 PPO epochs, 1,024-sample minibatches, learning rate `3e-4`, discount `0.99`, GAE lambda `0.95`, clip `0.2`, value coefficient `0.5`, entropy coefficient `0.001`, gradient clipping `0.5`, 1,048,576 transitions, and 128 PPO updates. Checkpoints include actor, critic, optimizer, learned action standard deviation, counters, configuration, best metrics, and practical Python/NumPy/PyTorch RNG state. Only compatible v2 checkpoints can resume; v1 architecture is explicitly rejected and is not migrated.

The deterministic lexicographic selection tuple is:

```text
(agency pass,
 release success rate,
 -median capped settling time,
 -mean final level error,
 -mean final pitch/roll rate,
 random-player mean survival,
 -random-player mean assist)
```

This selected the checkpoint evaluated at 8,192 transitions, not the final PPO update.

## Measured evaluation

The v1 baseline is preserved in [stability_assist_baseline_v1.json](stability_assist_baseline_v1.json). On its original 40-seed random-player suite, v1 measured 2.010 s no-assist, 2.139 s zero-actor, and 2.213 s trained mean survival; its level no-input sink rate was 0.468 m/s. V1 had no deliberate release suite, so these numbers are context rather than a like-for-like recovery comparison.

The final v2 report is [stability_assist_evaluation.json](stability_assist_evaluation.json), using evaluation seeds 4100–4139 with identical initial state, physics, `dt`, and button scripts for every mode.

### Release recovery (40 cases, 4.5 second fixed window)

Failures are assigned the 4.5 second window cap in settling-time statistics.

| Metric | No assist | PD-only / zero residual | Trained residual |
| --- | ---: | ---: | ---: |
| Recovery success | 0.0% | 62.5% | 62.5% |
| Mean settling time | 4.500 s | 3.482 s | 3.467 s |
| Median settling time | 4.500 s | 3.380 s | 3.364 s |
| P90 settling time | 4.500 s | 4.500 s | 4.500 s |
| Mean final tilt | 100.74° | 28.71° | 27.16° |
| Mean final pitch/roll rate | 1.059 | 0.376 | 0.371 rad/s |
| Mean maximum post-release tilt | 125.14° | 60.55° | 59.50° |
| Mean attitude overshoot | 100.41° | 35.82° | 34.93° |
| Crash before settling | 100.0% | 37.5% | 37.5% |
| Mean assist magnitude | 0 | 0.0919 | 0.0920 |
| Mean residual magnitude | 0 | 0 | 0.0745 |
| Cases with PD saturation | 0% | 77.5% | 77.5% |

The trained residual improves three meaningful metrics over PD-only without changing recovery success or agency: median settling is 0.016 s faster, final tilt is 1.55 degrees lower, and final rate is 0.005 rad/s lower.

### Random player, no-input sink, and agency

| Random-player metric | No assist | PD-only | Trained residual |
| --- | ---: | ---: | ---: |
| Mean survival | 1.837 s | 2.051 s | 2.060 s |
| Median survival | 1.725 s | 1.904 s | 1.881 s |
| Crash rate | 100% | 100% | 100% |
| Rotor-strike rate | 90.0% | 90.0% | 87.5% |
| Mean maximum tilt | 125.40° | 130.80° | 130.57° |
| Mean pitch/roll rate | 1.833 | 1.830 | 1.779 rad/s |
| Mean assist magnitude | 0 | 0.0294 | 0.0445 |

All eight moderate no-input cases recovered and then descended: mean recovery time 3.173 s, vertical velocity `-0.593 m/s`, settled tilt 3.76 degrees, settled rate 0.089 rad/s, zero altitude-increasing cases, and zero crashes.

The agency script passes. While buttons remain active, v2 reaches 24.61 degrees tilt and moves 0.126 m horizontally, essentially matching no assist. After release it settles and finishes at 5.04 degrees instead of crashing. A sustained single-motor script still reaches 176.1 degrees and crashes.

Known limitations: 37.5% of the deliberately difficult release cases cannot settle before ground/rotor impact or the fixed window, producing the capped P90. Random-player scripts are intentionally destructive and all eventually crash. Recovery is not guaranteed at low altitude, extreme attitude/rate, adverse motor saturation, or large player-induced momentum, and settings far outside the ±20% training domain are not established.

## Portable model, baking, and parity

`models/stability_assist.qwasmlp` is 8,868 bytes, little-endian, format version 2:

```text
8 bytes  magic "QWASMLP\0"
u32      format version (2)
u32 x 4  dimensions 32, 32, 32, 2
f32 x 32 observation scales
f32      layer 1 (32 x 32 weights, 32 biases)
f32      layer 2 (32 x 32 weights, 32 biases)
f32      layer 3 (2 x 32 weights, 2 biases)
```

The baked header is `include/generated/stability_assist_weights.h`, model ID `sha256-4caf3ded29099920`. Native and web builds use it by default with no runtime model file. Native `--assist-weights` still loads compatible v2 files; invalid or v1 files report a clear error and retain the baked fallback.

The committed fixtures test 32 inference cases, a 180-frame disabled/unassisted physics trajectory, and a 240-frame complete learned release trajectory. PyTorch versus exported inference differs by at most `5.96e-08`; C++ portable and baked outputs and Python/C++ trajectories pass the configured tolerance.

## Commands

```bash
python -m pip install -r tools/requirements-train.txt
python tools/train_stability_assist.py --preset smoke
python tools/train_stability_assist.py --preset train --output-dir checkpoints/v2_train
python tools/train_stability_assist.py --preset train --resume checkpoints/v2_train/stability_assist_train_latest.pt
python tools/evaluate_stability_assist.py --model models/stability_assist.qwasmlp
python tools/export_stability_assist.py --checkpoint checkpoints/v2_train/stability_assist_train_best.pt --output models/stability_assist.qwasmlp
python tools/bake_stability_assist_weights.py --input models/stability_assist.qwasmlp --output include/generated/stability_assist_weights.h
python tools/generate_stability_assist_parity.py --checkpoint checkpoints/v2_train/stability_assist_train_best.pt --model models/stability_assist.qwasmlp
python -m unittest discover -s tests -p "test_*.py" -v

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
./build/QWAS --assist-weights models/stability_assist.qwasmlp

emcmake cmake -S . -B build-web -DCMAKE_BUILD_TYPE=Release -DQWAS_BUILD_NATIVE=OFF -DQWAS_BUILD_WEB=ON
cmake --build build-web --target qwas_web
```

With a multi-configuration Windows generator, the native executable is `build/Release/QWAS.exe`.
