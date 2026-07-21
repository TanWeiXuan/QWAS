# Easy Mode stability assist

Easy Mode is an optional residual controller that helps arrest dangerous pitch and roll motion. It is not an autopilot: the player still commands all four motors and must create the tilt, lift, translation, yaw, and landing needed to finish the game. Easy Mode starts disabled on every application launch, is not persisted, and remains selected across retries only within the current process.

## Runtime thrust flow

`Drone::playerThrust` is the persistent result of the existing button ramp-up/ramp-down logic. `Drone::appliedThrust` is recomputed every frame and is the only thrust used by physics, rendering, rotor animation, and HUD bars:

```text
player thrust
+ equal below-hover top-up
+ pitch/roll-only learned differential
= applied thrust
```

The equal top-up targets `0.95 * mass * gravity` total thrust and is added only when the player's current total is lower:

```text
topUpPerMotor = max(0, 0.95 * mass * gravity - sum(playerThrust)) / 4
```

It never removes player thrust and remains below hover. With default physics, the measured no-input sink rate is 0.468 m/s over the latter half of a five-second test. Learned corrections cannot add collective thrust.
If an extreme settings profile makes the target physically infeasible, the equal top-up is limited by the motor with the least remaining headroom.

## Intervention and action constraints

The intervention gate is the maximum of two smoothstep gates:

- tilt: zero through 20 degrees, fully active at 60 degrees;
- combined body pitch/roll angular rate: zero through 1.5 rad/s, fully active at 5.0 rad/s.

The actor emits four `tanh` values. Runtime projects them onto the QWAS motor geometry:

```text
pitch = [ 1,  1, -1, -1]
roll  = [-1,  1, -1,  1]
```

The projected correction has exactly zero collective and yaw components. It is scaled by 0.35 times the current per-motor maximum thrust, Easy Mode strength, and the intervention gate. One common feasibility factor then keeps every final motor in `[0, maximum thrust]`; per-motor clipping is only a final numerical safeguard. The player can deliberately tilt because the gate is inactive for ordinary motion, and poor inputs can still cause a crash.

## Observation specification

Raw values are divided by the fixed scale below and clamped to `[-4, 4]`. The same scales are stored in `.qwasmlp`, baked into C++, and tested across Python and C++.

| Index | Value | Scale |
| ---: | --- | ---: |
| 0 | body-up world X | 1 |
| 1 | body-up world Y | 1 |
| 2 | body-up world Z | 1 |
| 3 | body angular velocity X (pitch) | 5 rad/s |
| 4 | body angular velocity Y (yaw) | 5 rad/s |
| 5 | body angular velocity Z (roll) | 5 rad/s |
| 6–9 | normalized player thrust: front-left, front-right, rear-left, rear-right | 1 |
| 10–13 | player buttons: front-left, front-right, rear-left, rear-right | 1 |
| 14 | frame `dt` | 1/60 s |
| 15 | intervention gate | 1 |
| 16 | mass | 0.5 kg |
| 17 | gravity | 9.81 m/s² |
| 18 | maximum thrust | 5 N |
| 19 | thrust ramp up | 10 N/s |
| 20 | thrust ramp down | 10 N/s |
| 21 | arm length | 0.25 m |
| 22 | pitch inertia | 0.4 kg·m² |
| 23 | yaw inertia | 0.7 kg·m² |
| 24 | roll inertia | 0.4 kg·m² |
| 25 | linear drag | 1 /s |
| 26 | angular drag | 0.5 /s |
| 27 | yaw coefficient | 0.02 |

The deployed actor is `28 -> tanh(32) -> tanh(32) -> tanh(4)`, with 2,116 weights and biases. The PPO critic and trainable diagonal Gaussian standard deviation exist only in checkpoints.

## Python dynamics and input generation

`tools/stability_assist/environment.py` is a NumPy-vectorized mirror of `Drone::Update`: the motor layout, torque sums, reactive yaw, body-frame angular acceleration, angular drag, quaternion update `q + 0.5 * (q * omega) * dt`, body-up thrust, gravity, linear drag, velocity, and position use the same order. A committed 180-step fixture compares it directly with the C++ `Drone` implementation.

Training randomizes `dt` from 1/75 to 1/45 seconds. Motor buttons are held for 0.1–0.8 seconds and sample zero, one, two, three, or four active motors, including deliberately unbalanced patterns. No expert flies toward the pad.

Initial altitude is 3–7 m. Most episodes begin within ±22 degrees pitch/roll and ±1.8 rad/s; 20% use up to ±55 degrees and ±4 rad/s. Yaw, linear velocity, initial player thrust, and horizontal offset are also randomized. Each physics parameter is independently varied by ±20%, while 35% of resets use the exact default profile. This intentionally does not cover the full extreme settings-slider range.

## Reward and PPO

Reward is the timestep-scaled sum of:

- `+1.0/s` alive;
- `-8.0` terminal crash;
- `-4.0 * excessTilt²` only beyond 35 degrees;
- `-0.15 * pitchRollRate²`;
- `-0.010 * assistMagnitude`;
- `-0.012 * assistChange`;
- `-0.020 * preGateActionMagnitude` while the gate is below 0.1.

There is no reward for forward progress, horizontal position/displacement, forward velocity, facing the goal, reaching or landing at the destination, altitude gain, or hovering. Position is used only for crash bounds, and horizontal displacement is reported only by the player-agency evaluation.

The final run used seed `20260721`, 64 environments, 128 rollout steps, four PPO epochs, 1,024-sample minibatches, learning rate `3e-4`, discount `0.99`, GAE lambda `0.95`, clip range `0.2`, value coefficient `0.5`, entropy coefficient `0.001`, gradient norm `0.5`, and 1,048,576 transitions (128 PPO updates). Training supports CPU/CUDA selection, resumable optimizer/model/RNG state, progress and checkpoint intervals, and deterministic best-checkpoint selection.

Sparse crash outcomes initially produced a nearly constant residual. The final reproducible run therefore starts the actor with 300 small supervised batches of a local pitch/roll damping target derived only from body-up and pitch/roll rate. PPO then optimizes that actor against the randomized survival task. This warm start contains no position, velocity, goal, altitude, player-progress, or navigation target.

## Evaluation

The committed report is [stability_assist_evaluation.json](stability_assist_evaluation.json). It uses seeds 4100–4139 and exactly the same initial state, physics, frame times, and held player-button sequence for every mode.

| Metric | No assist | Easy + zero actor | Easy + trained actor |
| --- | ---: | ---: | ---: |
| Mean survival | 2.010 s | 2.139 s | 2.213 s |
| Median survival | 1.829 s | 1.962 s | 2.020 s |
| Crash rate | 100% | 100% | 100% |
| Rotor-strike rate | 85.0% | 85.0% | 82.5% |
| Mean maximum tilt | 127.0° | 134.0° | 127.6° |
| Mean pitch/roll rate | 1.875 rad/s | 1.904 rad/s | 1.728 rad/s |
| Mean assist magnitude | 0 | 0 | 0.0602 |
| Gate-active fraction | 0.858 | 0.878 | 0.883 |
| Near-zero assist fraction | 1.000 | 1.000 | 0.572 |

The trained policy improves mean survival by 10.1% over no assist. The scripted agency test passes: Easy Mode reaches 179.4 degrees maximum tilt and moves 18.9 m horizontally, so it demonstrably does not hold the drone level or prevent player-driven movement. This intentionally aggressive script can still crash the drone.

Known limitations: every highly perturbed ten-second evaluation scenario eventually crashes, gains outside the ±20% training domain are not established, and this model reduces risk rather than guaranteeing recovery. The below-hover collective top-up accounts for part of the no-assist improvement; the trained differential additionally improves survival and lowers angular rate and rotor strikes.

## Portable model and native override

`models/stability_assist.qwasmlp` is 8,604 bytes, little-endian, and contains:

```text
8 bytes  magic "QWASMLP\0"
u32      format version (1)
u32 x 4  input, hidden-1, hidden-2, output dimensions
f32 x 28 observation scales
f32      layer 1 weights (32 x 28 row-major), then 32 biases
f32      layer 2 weights (32 x 32 row-major), then 32 biases
f32      layer 3 weights (4 x 32 row-major), then 4 biases
```

The C++ loader validates magic, version, exact dimensions, exact length, positive finite scales, and finite weights. Failed loads leave the baked actor active and print a warning. Native development can use `--assist-weights`; web builds always use `include/generated/stability_assist_weights.h`. Inference uses fixed-size arrays and `std::tanh`, with no per-frame allocation or inference dependency.

## Commands

```bash
python -m pip install -r tools/requirements-train.txt
python tools/train_stability_assist.py --preset smoke
python tools/train_stability_assist.py --preset train
python tools/train_stability_assist.py --preset train --resume checkpoints/stability_assist_train_latest.pt
python tools/evaluate_stability_assist.py --checkpoint checkpoints/stability_assist_train_best.pt
python tools/export_stability_assist.py --checkpoint checkpoints/stability_assist_train_best.pt --output models/stability_assist.qwasmlp
python tools/bake_stability_assist_weights.py --input models/stability_assist.qwasmlp --output include/generated/stability_assist_weights.h
python tools/generate_stability_assist_parity.py --checkpoint checkpoints/stability_assist_train_best.pt --model models/stability_assist.qwasmlp
python -m unittest tests.test_stability_assist -v

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
./build/QWAS --assist-weights models/stability_assist.qwasmlp

emcmake cmake -S . -B build-web -DCMAKE_BUILD_TYPE=Release -DQWAS_BUILD_NATIVE=OFF -DQWAS_BUILD_WEB=ON
cmake --build build-web --target qwas_web
```

With a multi-configuration Windows generator, the native executable is `build/Release/QWAS.exe`.
