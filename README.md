# QWAS — Quadrotor With Awkward Strokes

A 3D drone physics game inspired by [QWOP](https://www.foddy.net/Athletics.html). Instead of controlling a runner's legs, you control a quadrotor's four propellers individually. Coordinating four independent thrust sources to achieve stable flight is harder than it sounds.

## Gameplay

You control each of the drone's four motors with a single key. Hold a key to ramp up that motor's thrust; release it and the thrust drops quickly.

```
Q  W    ← front motors
A  S    ← rear motors
```

| Key | Motor | Color |
|-----|-------|-------|
| Q | Front-left | Red |
| W | Front-right | Blue |
| A | Rear-left | Green |
| S | Rear-right | Yellow |

**Goal:** fly from the green starting pad to the orange landing pad 25 metres ahead. Land gently — low speed, low tilt — to win.

**Crash conditions:** any rotor hits the ground (outside the starting pad).

**HUD:** four thrust bars at the bottom of the screen show each motor's current output in its colour.

### Tips

- Hover requires roughly equal thrust on all four motors (~41% of max each).
- Q+S and W+A are diagonal pairs — imbalancing them causes yaw rotation.
- The drone is intentionally difficult to stabilize; small corrections beat big ones.

## Controls

| Input | Action |
|-------|--------|
| Q / W / A / S | Hold to increase motor thrust; release to cut it |
| Space | Start game (from menu) |
| R | Restart (after crash or win) |
| Esc / window close | Quit |

## Building

**Requirements:** CMake 3.16+, a C++17 compiler, Git (for FetchContent). raylib is downloaded automatically — no manual dependency installation needed.

```bash
# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)   # Linux/macOS
cmake --build build              # Windows

# Run
./build/QWAS        # Linux/macOS
build\QWAS.exe      # Windows
```

The first build fetches and compiles raylib (~1–2 minutes). Subsequent builds are fast.

### Platform notes

- **macOS:** Cocoa, IOKit, and OpenGL frameworks are linked automatically.
- **Linux:** requires `libgl1-mesa-dev`, `libx11-dev`, and related X11/GL headers (standard raylib dependencies).
- **CMake 4.x:** a compatibility shim is included for `cmake_minimum_required` version policy changes introduced in CMake 4.0.

## Project structure

```
QWAS/
├── CMakeLists.txt       — build definition; fetches raylib 6.0
├── include/
│   ├── drone.h          — Drone struct, physics constants, rotor API
│   └── game.h           — Game struct, state machine, camera constants
├── src/
│   ├── main.cpp         — window init and game loop
│   ├── drone.cpp        — 3D rigid-body physics and rendering
│   └── game.cpp         — game logic, camera, world, collision detection
└── third_party/         — placeholder; raylib lands in build/_deps/
```

## Physics

The drone is simulated as a rigid body with full 3D rotation:

- **Orientation** is tracked as a quaternion and integrated each frame using body-frame angular velocity.
- **Thrust** from each motor creates torques about the pitch and roll axes; imbalanced diagonal pairs create yaw torque.
- **Linear drag** and **angular drag** provide passive damping.

Physics constants (mass, inertia, drag, thrust limits) live in `include/drone.h` and are easy to tune.
