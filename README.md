# QWAS - Quadrotor With Awkward Strokes

![App Screenshot](doc/qwas.png)

QWAS is a small 3D drone physics game inspired by QWOP. Instead of controlling a runner's legs, you control a quadrotor's four propellers independently. Coordinating four thrust sources into stable flight is harder than it sounds.

[Try it here now!](https://chiaxr.github.io/QWAS/)

## Gameplay

Hold a motor input to ramp up that motor's thrust; release it and the thrust drops quickly.

```text
Q  W    front motors
A  S    rear motors
```

| Input | Motor | Color |
| --- | --- | --- |
| Q / top-left quadrant | Front-left | Red |
| W / top-right quadrant | Front-right | Blue |
| A / bottom-left quadrant | Rear-left | Green |
| S / bottom-right quadrant | Rear-right | Yellow |

Goal: fly from the green starting pad to the orange landing pad 25 meters ahead. Land gently with low speed and low tilt to win.

Crash condition: any rotor hits the ground outside the starting or landing pad.

## Controls

| Input | Action |
| --- | --- |
| Q / W / A / S | Hold to increase motor thrust; release to cut it |
| Touch/click quadrant | Hold a screen quadrant to control the matching motor |
| Space | Start from the menu |
| Tap/click | Start from the menu, retry after a crash delay, or restart after a win |
| R | Restart after crash or win |
| Tab | Physics settings |
| Arrow keys | Adjust physics settings |
| Backspace | Leave physics settings |
| Esc / window close | Quit native desktop build |

The web version is landscape-only. In portrait orientation the canvas is hidden, the game is paused, and a rotate prompt is shown until the viewport returns to landscape.

## Native Build

Requirements: CMake 3.16+, a C++17 compiler, and Git. raylib 6.0 is fetched automatically by CMake.

The native desktop commands are unchanged:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Run the native executable:

```bash
./build/QWAS
```

On Windows with Visual Studio generators, the executable is typically under:

```powershell
build\Debug\QWAS.exe
```

Platform notes:

- macOS native builds link Cocoa, IOKit, and OpenGL frameworks.
- Linux native builds use the normal raylib desktop dependencies such as OpenGL and X11/Wayland development packages.
- The desktop build does not depend on Emscripten, JavaScript, or web files.

## Local Emscripten Build

Install and activate Emscripten first, then configure with `emcmake`:

```bash
emcmake cmake -S . -B build-web \
  -DCMAKE_BUILD_TYPE=Release \
  -DQWAS_BUILD_NATIVE=OFF \
  -DQWAS_BUILD_WEB=ON

cmake --build build-web --target qwas_web
```

Expected outputs:

```text
build-web/dist/index.html
build-web/dist/index.js
build-web/dist/index.wasm
```

Serve locally with:

```bash
emrun build-web/dist/index.html
```

You can also serve `build-web/dist` with any static file server.

## Project Structure

```text
QWAS/
|-- CMakeLists.txt
|-- .github/workflows/pages.yml
|-- include/
|   |-- drone.h
|   |-- game.h
|   `-- qwas_app.h
|-- src/
|   |-- drone.cpp
|   |-- game.cpp
|   |-- main.cpp
|   |-- qwas_app.cpp
|   `-- web/
|       `-- web_main.cpp
|-- web/
|   `-- emscripten_shell.html
`-- doc/
    `-- qwas.png
```

`qwas_game` contains the shared drone, game, and application code. `QWAS` is the native launcher. `qwas_web` is the Emscripten launcher and uses `emscripten_set_main_loop()`.
