#include "game.h"
#include "raymath.h"
#include <cmath>

// ---------------------------------------------------------------------------
//  Lifecycle
// ---------------------------------------------------------------------------

void Game::Init() {
    screenWidth  = 1280;
    screenHeight = 720;

    startPad.position = {0, 0, 0};
    startPad.halfSize = 1.0f;
    pad.position      = {0, 0, PAD_WORLD_Z};
    pad.halfSize      = 1.0f;
    bestScore         = 0;

    camera.fovy       = CAMERA_FOV;
    camera.projection = CAMERA_PERSPECTIVE;
    camera.up         = {0, 1, 0};

    // Camera starts looking at spawn from behind/above
    camera.position = {0, 4, 8};
    camera.target   = {0, DRONE_REST_Y, 0};

    state = GameState::MENU;
    drone.Init({0, DRONE_REST_Y, 0});
    deadTimer = 0;
    settingsSelectedIdx = 0;
}

void Game::Reset() {
    drone.Init({0, DRONE_REST_Y, 0});
    deadTimer = 0;

    camera.position = {0, 4, 8};
    camera.target   = {0, DRONE_REST_Y, 0};
    camera.up       = {0, 1, 0};
    camera.fovy     = CAMERA_FOV;
}

// ---------------------------------------------------------------------------
//  Update
// ---------------------------------------------------------------------------

void Game::Update(float dt) {
    switch (state) {
        case GameState::MENU:     UpdateMenu(dt);     break;
        case GameState::PLAYING:  UpdatePlaying(dt);  break;
        case GameState::DEAD:     UpdateDead(dt);     break;
        case GameState::WIN:      UpdateWin(dt);      break;
        case GameState::SETTINGS: UpdateSettings(dt); break;
    }
}

void Game::UpdateMenu(float dt) {
    (void)dt;
    if (IsKeyPressed(KEY_SPACE)) {
        state = GameState::PLAYING;
    }
    if (IsKeyPressed(KEY_TAB)) {
        settingsSelectedIdx = 0;
        state = GameState::SETTINGS;
    }
}

void Game::UpdateSettings(float dt) {
    (void)dt;

    struct Entry { float* val; float minV; float maxV; float step; };
    Entry entries[] = {
        { &DRONE_MASS,       0.1f,  5.0f,  0.05f  },
        { &GRAVITY,          0.0f,  20.0f, 0.1f   },
        { &MAX_THRUST,       0.5f,  10.0f, 0.1f   },
        { &THRUST_RAMP_UP,   1.0f,  20.0f, 0.5f   },
        { &THRUST_RAMP_DOWN, 1.0f,  30.0f, 0.5f   },
        { &ARM_LENGTH,       0.05f, 1.0f,  0.01f  },
        { &I_PITCH,          0.001f,0.05f, 0.001f },
        { &I_YAW,            0.001f,0.05f, 0.001f },
        { &I_ROLL,           0.001f,0.05f, 0.001f },
        { &LIN_DRAG,         0.0f,  3.0f,  0.05f  },
        { &ANG_DRAG,         0.0f,  10.0f, 0.1f   },
        { &K_YAW,            0.001f,0.2f,  0.001f },
    };
    constexpr int COUNT = 12;

    if (IsKeyPressed(KEY_BACKSPACE)) { state = GameState::MENU; return; }

    if (IsKeyPressed(KEY_R)) {
        DRONE_MASS       = DRONE_MASS_DEFAULT;
        GRAVITY          = GRAVITY_DEFAULT;
        MAX_THRUST       = MAX_THRUST_DEFAULT;
        THRUST_RAMP_UP   = THRUST_RAMP_UP_DEFAULT;
        THRUST_RAMP_DOWN = THRUST_RAMP_DOWN_DEFAULT;
        ARM_LENGTH       = ARM_LENGTH_DEFAULT;
        I_PITCH          = I_PITCH_DEFAULT;
        I_YAW            = I_YAW_DEFAULT;
        I_ROLL           = I_ROLL_DEFAULT;
        LIN_DRAG         = LIN_DRAG_DEFAULT;
        ANG_DRAG         = ANG_DRAG_DEFAULT;
        K_YAW            = K_YAW_DEFAULT;
        return;
    }

    if (IsKeyPressed(KEY_UP))   settingsSelectedIdx = (settingsSelectedIdx - 1 + COUNT) % COUNT;
    if (IsKeyPressed(KEY_DOWN)) settingsSelectedIdx = (settingsSelectedIdx + 1) % COUNT;

    Entry& e = entries[settingsSelectedIdx];
    if (IsKeyDown(KEY_LEFT))  *e.val = fmaxf(e.minV, *e.val - e.step);
    if (IsKeyDown(KEY_RIGHT)) *e.val = fminf(e.maxV, *e.val + e.step);
}

void Game::UpdatePlaying(float dt) {
    if (IsKeyPressed(KEY_R)) { Reset(); return; }

    drone.SetRotorInput(ROTOR_FRONT_LEFT, IsKeyDown(KEY_Q), dt);
    drone.SetRotorInput(ROTOR_FRONT_RIGHT, IsKeyDown(KEY_W), dt);
    drone.SetRotorInput(ROTOR_REAR_LEFT, IsKeyDown(KEY_A), dt);
    drone.SetRotorInput(ROTOR_REAR_RIGHT, IsKeyDown(KEY_S), dt);

    drone.Update(dt);
    UpdateCamera(dt);
    CheckCollisions();

    float progress = fminf(drone.distanceTraveled / fabsf(PAD_WORLD_Z) * 100.0f, 100.0f);
    if (progress > bestScore) bestScore = progress;
}

void Game::UpdateDead(float dt) {
    deadTimer -= dt;
    if (deadTimer <= 0.0f && IsKeyPressed(KEY_R)) {
        Reset();
        state = GameState::PLAYING;
    }
}

void Game::UpdateWin(float dt) {
    (void)dt;
    if (IsKeyPressed(KEY_R)) {
        Reset();
        state = GameState::PLAYING;
    }
}

void Game::UpdateCamera(float dt) {
    Vector3 fwd = drone.GetForwardDir();

    Vector3 targetPos = {
        drone.position.x - fwd.x * CHASE_DIST,
        drone.position.y + CHASE_HEIGHT,
        drone.position.z - fwd.z * CHASE_DIST
    };

    float s = CAMERA_SMOOTH * dt;
    camera.position.x += (targetPos.x - camera.position.x) * s;
    camera.position.y += (targetPos.y - camera.position.y) * s;
    camera.position.z += (targetPos.z - camera.position.z) * s;

    camera.target = {
        drone.position.x + fwd.x * 0.5f,
        drone.position.y,
        drone.position.z + fwd.z * 0.5f
    };
    camera.up = {0, 1, 0};
}

void Game::CheckCollisions() {
    // --- Starting pad: ground contact here is never a crash ---
    bool onStartPad = fabsf(drone.position.x - startPad.position.x) < startPad.halfSize &&
                      fabsf(drone.position.z - startPad.position.z) < startPad.halfSize;

    if (onStartPad) {
        // Detect contact: drone sinking into the pad surface
        bool grounded = drone.position.y < DRONE_REST_Y;

        if (grounded) {
            // Ground reaction: push back up and kill downward velocity
            drone.position.y = DRONE_REST_Y;
            if (drone.velocity.y < 0.0f) drone.velocity.y = 0.0f;
            // Heavy angular damping while on the pad (simulates contact friction)
            drone.angularVel.x *= 0.85f;
            drone.angularVel.z *= 0.85f;
        }
        return;  // never crash while on the starting pad
    }

    // --- Normal crash checks (only when away from starting pad) ---
    for (int i = 0; i < ROTOR_COUNT; i++) {
        if (drone.GetRotorWorldPos((RotorID)i).y < 0.0f) {
            drone.alive = false;
            state       = GameState::DEAD;
            deadTimer   = 1.5f;
            return;
        }
    }

    if (drone.position.y < 0.0f) {
        drone.alive = false;
        state       = GameState::DEAD;
        deadTimer   = 1.5f;
        return;
    }

    if (drone.position.y > 15.0f || fabsf(drone.position.x) > 20.0f) {
        drone.alive = false;
        state       = GameState::DEAD;
        deadTimer   = 1.5f;
        return;
    }

    // --- Win: gentle landing on destination pad ---
    bool overPad = fabsf(drone.position.x - pad.position.x) < pad.halfSize &&
                   fabsf(drone.position.z - pad.position.z) < pad.halfSize;
    float speed  = Vector3Length(drone.velocity);
    bool gentle  = drone.position.y < 0.5f && speed < 2.0f && drone.GetTiltAngle() < 20.0f;

    if (overPad && gentle) {
        state     = GameState::WIN;
        bestScore = 100.0f;
    }
}

// ---------------------------------------------------------------------------
//  Draw
// ---------------------------------------------------------------------------

void Game::Draw() const {
    BeginDrawing();
    ClearBackground(SKYBLUE);

    BeginMode3D(camera);
    DrawWorld();
    if (state != GameState::MENU) drone.Draw();
    EndMode3D();

    DrawOverlay();
    EndDrawing();
}

void Game::DrawOverlay() const {
    switch (state) {
        case GameState::MENU:     DrawMenu();     break;
        case GameState::PLAYING:  DrawPlaying();  break;
        case GameState::DEAD:     DrawDead();     break;
        case GameState::WIN:      DrawWin();      break;
        case GameState::SETTINGS: DrawSettings(); break;
    }
}

void Game::DrawWorld() const {
    // Ground plane slightly below grid to avoid Z-fighting
    DrawPlane({0, -0.01f, 0}, {100.0f, 100.0f}, DARKGREEN);
    DrawGrid(60, 1.0f);

    // Landing pad
    float padX = pad.position.x, padZ = pad.position.z, half = pad.halfSize;
    DrawCube({padX, 0.05f, padZ}, half * 2, 0.10f, half * 2, ORANGE);
    DrawCubeWires({padX, 0.05f, padZ}, half * 2, 0.10f, half * 2, DARKBROWN);

    // "H" marker on pad top using lines
    float top = 0.12f;
    float hw  = half * 0.55f;  // horizontal bar half-width
    float hh  = half * 0.70f;  // vertical bar half-height
    DrawLine3D({padX - hw, top, padZ - hh}, {padX - hw, top, padZ + hh}, WHITE);
    DrawLine3D({padX + hw, top, padZ - hh}, {padX + hw, top, padZ + hh}, WHITE);
    DrawLine3D({padX - hw, top, padZ      }, {padX + hw, top, padZ      }, WHITE);

    // Starting pad (lime green, same size as destination pad)
    float sX = startPad.position.x, sZ = startPad.position.z, sH = startPad.halfSize;
    DrawCube({sX, 0.05f, sZ}, sH * 2, 0.10f, sH * 2, LIME);
    DrawCubeWires({sX, 0.05f, sZ}, sH * 2, 0.10f, sH * 2, DARKGREEN);
    // Arrow pointing toward destination pad
    DrawLine3D({sX,       0.12f, sZ - sH * 0.3f}, {sX,       0.12f, sZ - sH * 0.7f}, WHITE);
    DrawLine3D({sX - 0.2f, 0.12f, sZ - sH * 0.4f}, {sX, 0.12f, sZ - sH * 0.7f}, WHITE);
    DrawLine3D({sX + 0.2f, 0.12f, sZ - sH * 0.4f}, {sX, 0.12f, sZ - sH * 0.7f}, WHITE);

    // Reference trees along path at x=±4, every 5m forward
    for (int i = 1; i <= 5; i++) {
        float z = -i * 4.5f;
        const float treeXs[] = {-4.0f, 4.0f};
        for (float x : treeXs) {
            DrawCylinder({x, 0, z}, 0.15f, 0.0f, 2.0f, 6, BROWN);
            DrawSphere({x, 2.2f, z}, 0.65f, DARKGREEN);
        }
    }
}

// ---------------------------------------------------------------------------
//  State overlays (2D HUD, all drawn after EndMode3D)
// ---------------------------------------------------------------------------

static void DrawCenteredText(const char* text, int y, int fontSize, Color color) {
    int w = MeasureText(text, fontSize);
    DrawText(text, (GetScreenWidth() - w) / 2, y, fontSize, color);
}

void Game::DrawMenu() const {
    // Semi-transparent backdrop in centre
    int bw = 700, bh = 480;
    int bx = (screenWidth - bw) / 2, by = (screenHeight - bh) / 2 - 30;
    DrawRectangle(bx, by, bw, bh, {0, 0, 0, 160});

    DrawCenteredText("QWAS", by + 20, 72, WHITE);
    DrawCenteredText("Quadrotor With Awkward Strokes", by + 100, 22, LIGHTGRAY);

    // Controls
    const char* lines[] = {
        "Hold Q W A S to spin each propeller independently.",
        "Longer hold = more thrust. Release = thrust drops fast.",
        "",
        "Fly from the white spawn pad to the orange landing pad.",
        "Land gently (low speed, nearly level) to win.",
        "",
        "SPACE  -  Start",
        "R      -  Restart at any time",
        "TAB    -  Physics Settings"
    };
    int lineY = by + 150;
    for (const char* line : lines) {
        DrawCenteredText(line, lineY, 18, WHITE);
        lineY += 28;
    }

    // Motor colour legend in keyboard layout
    struct { const char* key; Color col; const char* desc; } legend[4] = {
        {"Q", RED,    "Front-Left"},
        {"W", BLUE,   "Front-Right"},
        {"A", GREEN,  "Rear-Left"},
        {"S", YELLOW, "Rear-Right"}
    };
    int lx = bx + 60, ly = by + bh - 90;
    DrawText("Motors:", lx, ly, 18, LIGHTGRAY);
    for (int i = 0; i < 4; i++) {
        int ix = lx + (i % 2) * 290;
        int iy = ly + 26 + (i / 2) * 24;
        DrawRectangle(ix, iy, 16, 16, legend[i].col);
        DrawText(TextFormat("%s = %s", legend[i].key, legend[i].desc), ix + 22, iy, 17, WHITE);
    }
}

void Game::DrawPlaying() const {
    drone.DrawHUDBars(screenWidth, screenHeight);

    // Stats (top-right)
    int sx = screenWidth - 210, sy = 16;
    DrawText(TextFormat("Alt:      %.1f m",  drone.GetAltitude()),              sx, sy,      20, WHITE);
    DrawText(TextFormat("Speed:  %.1f m/s",  Vector3Length(drone.velocity)),    sx, sy + 26, 20, WHITE);
    DrawText(TextFormat("Progress: %.0f%%",
        fminf(drone.distanceTraveled / fabsf(PAD_WORLD_Z) * 100.0f, 100.0f)), sx, sy + 52, 20, YELLOW);
    if (bestScore > 0)
        DrawText(TextFormat("Best:   %.0f%%", bestScore), sx, sy + 78, 20, GREEN);

    float tilt = drone.GetTiltAngle();
    Color tc   = tilt < 15 ? GREEN : (tilt < 35 ? YELLOW : RED);
    DrawText(TextFormat("Tilt:     %.0f°",  tilt), sx, sy + 104, 20, tc);
}

void Game::DrawDead() const {
    drone.DrawHUDBars(screenWidth, screenHeight);

    if (deadTimer <= 0.0f) {
        DrawRectangle(0, 0, screenWidth, screenHeight, {0, 0, 0, 120});

        DrawCenteredText("CRASHED!", screenHeight / 2 - 80, 60, RED);

        float progress = fminf(drone.distanceTraveled / fabsf(PAD_WORLD_Z) * 100.0f, 100.0f);
        DrawCenteredText(TextFormat("Progress: %.0f%%", progress), screenHeight / 2, 30, WHITE);
        DrawCenteredText("Press R to retry", screenHeight / 2 + 50, 24, LIGHTGRAY);
    }
}

void Game::DrawWin() const {
    DrawRectangle(0, 0, screenWidth, screenHeight, {0, 60, 0, 150});

    DrawCenteredText("LANDED!",             screenHeight / 2 - 100, 80, GREEN);
    DrawCenteredText("Progress:  100%",     screenHeight / 2 - 10,  30, WHITE);
    DrawCenteredText("Perfect landing!",    screenHeight / 2 + 30,  24, GOLD);
    DrawCenteredText("Press R to fly again", screenHeight / 2 + 70, 24, LIGHTGRAY);

    drone.DrawHUDBars(screenWidth, screenHeight);
}

void Game::DrawSettings() const {
    struct Entry { const char* label; const char* unit; float* val; float minV; float maxV; };
    Entry entries[] = {
        { "Mass",            "kg",     &DRONE_MASS,       0.1f,  5.0f  },
        { "Gravity",         "m/s\xb2",&GRAVITY,          0.0f,  20.0f },
        { "Max Thrust",      "N",      &MAX_THRUST,       0.5f,  10.0f },
        { "Thrust Ramp Up",  "N/s",    &THRUST_RAMP_UP,   1.0f,  20.0f },
        { "Thrust Ramp Down","N/s",    &THRUST_RAMP_DOWN, 1.0f,  30.0f },
        { "Arm Length",      "m",      &ARM_LENGTH,       0.05f, 1.0f  },
        { "Inertia Pitch",   "kg\xb7m\xb2",&I_PITCH,     0.001f,0.05f },
        { "Inertia Yaw",     "kg\xb7m\xb2",&I_YAW,       0.001f,0.05f },
        { "Inertia Roll",    "kg\xb7m\xb2",&I_ROLL,       0.001f,0.05f },
        { "Linear Drag",     "/s",     &LIN_DRAG,         0.0f,  3.0f  },
        { "Angular Drag",    "/s",     &ANG_DRAG,         0.0f,  10.0f },
        { "Yaw Coeff",       "",       &K_YAW,            0.001f,0.2f  },
    };
    constexpr int COUNT = 12;
    constexpr int ROW_H = 34;

    int bw = 820, bh = 60 + COUNT * ROW_H + 50;
    int bx = (screenWidth  - bw) / 2;
    int by = (screenHeight - bh) / 2;
    DrawRectangle(bx, by, bw, bh, {0, 0, 0, 180});

    DrawCenteredText("PHYSICS SETTINGS", by + 14, 28, WHITE);

    int sliderX  = bx + 290;
    int sliderW  = 300;
    int labelX   = bx + 20;
    int valueX   = bx + bw - 120;

    for (int i = 0; i < COUNT; i++) {
        int ry = by + 54 + i * ROW_H;
        Entry& e = entries[i];

        if (i == settingsSelectedIdx)
            DrawRectangle(bx + 2, ry, bw - 4, ROW_H - 2, {255, 255, 255, 30});

        Color textCol = (i == settingsSelectedIdx) ? WHITE : LIGHTGRAY;

        // Label
        DrawText(e.label, labelX, ry + 8, 18, textCol);

        // Slider track
        DrawRectangle(sliderX, ry + 12, sliderW, 10, {80, 80, 80, 255});

        // Slider fill
        float t = (*e.val - e.minV) / (e.maxV - e.minV);
        int fillW = (int)(t * sliderW);
        Color fillCol = (i == settingsSelectedIdx) ? WHITE : GRAY;
        DrawRectangle(sliderX, ry + 12, fillW, 10, fillCol);

        // Slider thumb
        DrawRectangle(sliderX + fillW - 3, ry + 8, 6, 18,
                      (i == settingsSelectedIdx) ? YELLOW : DARKGRAY);

        // Value + unit
        DrawText(TextFormat("%.3g %s", *e.val, e.unit), valueX, ry + 8, 17, textCol);
    }

    DrawCenteredText("Up/Down: Select   Left/Right: Adjust   R: Reset defaults   Backspace: Back",
                     by + bh - 26, 16, {180, 180, 180, 255});
}
