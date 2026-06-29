#include "game.h"
#include "raymath.h"
#include <cmath>

namespace {
constexpr float TOUCH_GUIDE_FADE_SPEED = 1.8f;
constexpr float TAP_MAX_MOVE = 28.0f;

Rectangle GetRotorTouchZone(RotorID id, int screenW, int screenH) {
    float zoneW = screenW * 0.5f;
    float zoneH = screenH * 0.5f;

    switch (id) {
        case ROTOR_FRONT_LEFT:
            return {0.0f, 0.0f, zoneW, zoneH};
        case ROTOR_FRONT_RIGHT:
            return {screenW - zoneW, 0.0f, zoneW, zoneH};
        case ROTOR_REAR_LEFT:
            return {0.0f, screenH - zoneH, zoneW, zoneH};
        case ROTOR_REAR_RIGHT:
            return {screenW - zoneW, screenH - zoneH, zoneW, zoneH};
    }

    return {0.0f, 0.0f, 0.0f, 0.0f};
}

bool IsRotorTouchDown(RotorID id, int screenW, int screenH) {
    Rectangle zone = GetRotorTouchZone(id, screenW, screenH);

    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && CheckCollisionPointRec(GetMousePosition(), zone))
        return true;

    int touchCount = GetTouchPointCount();
    for (int i = 0; i < touchCount; i++) {
        if (CheckCollisionPointRec(GetTouchPosition(i), zone))
            return true;
    }

    return false;
}

bool IsPrimaryPointerDown() {
    return IsMouseButtonDown(MOUSE_BUTTON_LEFT) || GetTouchPointCount() > 0;
}

Vector2 GetPrimaryPointerPosition() {
    if (GetTouchPointCount() > 0)
        return GetTouchPosition(0);

    return GetMousePosition();
}

void DrawRotorTouchZone(RotorID id, const Drone& drone, int screenW, int screenH, const char* label, float alpha) {
    Rectangle zone = GetRotorTouchZone(id, screenW, screenH);
    Color color = drone.rotors[id].color;
    bool active = drone.rotors[id].thrust > 0.01f;

    DrawRectangleRec(zone, Fade(color, alpha * (active ? 0.20f : 0.06f)));
    DrawRectangleLinesEx(zone, active ? 4.0f : 2.0f, Fade(color, alpha * (active ? 0.80f : 0.35f)));

    int fontSize = 26;
    int labelW = MeasureText(label, fontSize);
    DrawText(label,
             (int)(zone.x + (zone.width - labelW) * 0.5f),
             (int)(zone.y + (zone.height - fontSize) * 0.5f),
             fontSize,
             Fade(color, alpha * 0.90f));
}

const char* GetCrashTitle(CrashReason reason) {
    switch (reason) {
        case CrashReason::ROTOR_STRIKE:   return "ROTOR STRIKE!";
        case CrashReason::GROUND_IMPACT:  return "GROUND IMPACT!";
        case CrashReason::TOO_HIGH:       return "TOO HIGH!";
        case CrashReason::OUT_OF_BOUNDS:  return "OUT OF BOUNDS!";
        case CrashReason::NONE:           return "CRASHED!";
    }

    return "CRASHED!";
}

const char* GetCrashHint(CrashReason reason) {
    switch (reason) {
        case CrashReason::ROTOR_STRIKE:   return "A rotor hit the ground.";
        case CrashReason::GROUND_IMPACT:  return "The drone body hit the ground.";
        case CrashReason::TOO_HIGH:       return "You flew above the safe altitude.";
        case CrashReason::OUT_OF_BOUNDS:  return "You left the flight area.";
        case CrashReason::NONE:           return "";
    }

    return "";
}
}  // namespace

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
    crashReason = CrashReason::NONE;
    drone.Init({0, DRONE_REST_Y, 0});
    deadTimer = 0;
    winTimer  = 0;
    settingsSelectedIdx = 0;
    touchGuideAlpha = 1.0f;
    touchGuideDismissed = false;
    tapWasDown = false;
    tapCandidate = false;
    tapStart = {0, 0};
}

void Game::Reset() {
    drone.Init({0, DRONE_REST_Y, 0});
    crashReason = CrashReason::NONE;
    deadTimer = 0;
    winTimer  = 0;
    touchGuideAlpha = 1.0f;
    touchGuideDismissed = false;
    tapWasDown = false;
    tapCandidate = false;
    tapStart = {0, 0};

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
    if (IsKeyPressed(KEY_SPACE) || ConsumeCompletedTap()) {
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
        { &I_PITCH,          0.01f, 1.0f,  0.05f  },
        { &I_YAW,            0.01f, 1.0f,  0.05f  },
        { &I_ROLL,           0.01f, 1.0f,  0.05f  },
        { &LIN_DRAG,         0.0f,  3.0f,  0.5f   },
        { &ANG_DRAG,         0.0f,  10.0f, 0.1f   },
        { &K_YAW,            0.01f, 0.2f,  0.01f  },
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

    int w = GetScreenWidth();
    int h = GetScreenHeight();

    bool frontLeftInput  = IsKeyDown(KEY_Q) || IsRotorTouchDown(ROTOR_FRONT_LEFT, w, h);
    bool frontRightInput = IsKeyDown(KEY_W) || IsRotorTouchDown(ROTOR_FRONT_RIGHT, w, h);
    bool rearLeftInput   = IsKeyDown(KEY_A) || IsRotorTouchDown(ROTOR_REAR_LEFT, w, h);
    bool rearRightInput  = IsKeyDown(KEY_S) || IsRotorTouchDown(ROTOR_REAR_RIGHT, w, h);

    if (frontLeftInput || frontRightInput || rearLeftInput || rearRightInput)
        touchGuideDismissed = true;

    if (touchGuideDismissed)
        touchGuideAlpha = fmaxf(0.0f, touchGuideAlpha - TOUCH_GUIDE_FADE_SPEED * dt);

    drone.SetRotorInput(ROTOR_FRONT_LEFT, frontLeftInput, dt);
    drone.SetRotorInput(ROTOR_FRONT_RIGHT, frontRightInput, dt);
    drone.SetRotorInput(ROTOR_REAR_LEFT, rearLeftInput, dt);
    drone.SetRotorInput(ROTOR_REAR_RIGHT, rearRightInput, dt);

    drone.Update(dt);
    UpdateCamera(dt);
    CheckCollisions();

    float progress = fminf(drone.distanceTraveled / fabsf(PAD_WORLD_Z) * 100.0f, 100.0f);
    if (progress > bestScore) bestScore = progress;
}

void Game::UpdateDead(float dt) {
    deadTimer -= dt;
    if (deadTimer <= 0.0f && (IsKeyPressed(KEY_R) || ConsumeCompletedTap())) {
        Reset();
        state = GameState::PLAYING;
    }
}

void Game::UpdateWin(float dt) {
    winTimer += dt;
    if (IsKeyPressed(KEY_R) || ConsumeCompletedTap()) {
        Reset();
        state = GameState::PLAYING;
    }
}

bool Game::ConsumeCompletedTap() {
    bool pointerDown = IsPrimaryPointerDown();
    bool completedTap = false;

    if (pointerDown) {
        Vector2 pointerPosition = GetPrimaryPointerPosition();
        if (!tapWasDown) {
            tapCandidate = true;
            tapStart = pointerPosition;
        } else if (tapCandidate && Vector2Distance(tapStart, pointerPosition) > TAP_MAX_MOVE) {
            tapCandidate = false;
        }
    } else if (tapWasDown && tapCandidate) {
        completedTap = true;
        tapCandidate = false;
    }

    tapWasDown = pointerDown;
    if (!pointerDown)
        tapCandidate = false;

    return completedTap || IsGestureDetected(GESTURE_TAP);
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

    // --- Destination pad: ground reaction + win check (before crash checks) ---
    bool overPad = fabsf(drone.position.x - pad.position.x) < pad.halfSize &&
                   fabsf(drone.position.z - pad.position.z) < pad.halfSize;
    if (overPad) {
        if (drone.position.y < DRONE_REST_Y) {
            drone.position.y = DRONE_REST_Y;
            if (drone.velocity.y < 0.0f) drone.velocity.y = 0.0f;
            drone.angularVel.x *= 0.85f;
            drone.angularVel.z *= 0.85f;
        }
        float speed = Vector3Length(drone.velocity);
        bool gentle = drone.position.y < 0.5f && speed < 2.0f && drone.GetTiltAngle() < 20.0f;
        if (gentle) {
            state     = GameState::WIN;
            bestScore = 100.0f;
        }
        return;
    }

    // --- Normal crash checks (only when away from both pads) ---
    for (int i = 0; i < ROTOR_COUNT; i++) {
        if (drone.GetRotorWorldPos((RotorID)i).y < 0.0f) {
            drone.alive = false;
            crashReason = CrashReason::ROTOR_STRIKE;
            state       = GameState::DEAD;
            deadTimer   = 1.5f;
            return;
        }
    }

    if (drone.position.y < 0.0f) {
        drone.alive = false;
        crashReason = CrashReason::GROUND_IMPACT;
        state       = GameState::DEAD;
        deadTimer   = 1.5f;
        return;
    }

    if (drone.position.y > 15.0f) {
        drone.alive = false;
        crashReason = CrashReason::TOO_HIGH;
        state       = GameState::DEAD;
        deadTimer   = 1.5f;
        return;
    }

    if (fabsf(drone.position.x) > 20.0f) {
        drone.alive = false;
        crashReason = CrashReason::OUT_OF_BOUNDS;
        state       = GameState::DEAD;
        deadTimer   = 1.5f;
        return;
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
    int bw = 760, bh = 540;
    int bx = (screenWidth - bw) / 2, by = (screenHeight - bh) / 2 - 30;
    DrawRectangle(bx, by, bw, bh, {0, 0, 0, 160});

    DrawCenteredText("QWAS", by + 20, 72, WHITE);
    DrawCenteredText("Quadrotor With Awkward Strokes", by + 100, 22, LIGHTGRAY);

    // Controls
    const char* lines[] = {
        "Keyboard: hold Q W A S to spin each propeller.",
        "Touch/click: hold a screen quadrant for the matching motor.",
        "Top: Q W    Bottom: A S",
        "Longer hold = more thrust. Release = thrust drops fast.",
        "",
        "Fly from the white spawn pad to the orange landing pad.",
        "Land gently (low speed, nearly level) to win.",
        "",
        "SPACE or tap  -  Start",
        "R or tap      -  Restart after crash/win",
        "TAB           -  Physics Settings (keyboard only)"
    };
    int lineY = by + 150;
    for (const char* line : lines) {
        DrawCenteredText(line, lineY, 18, WHITE);
        lineY += 24;
    }

    // Motor colour legend in keyboard layout
    struct { const char* key; Color col; const char* desc; } legend[4] = {
        {"Q", RED,    "Front-Left"},
        {"W", BLUE,   "Front-Right"},
        {"A", GREEN,  "Rear-Left"},
        {"S", YELLOW, "Rear-Right"}
    };
    int lx = bx + 60, ly = by + bh - 86;
    DrawText("Motors:", lx, ly, 18, LIGHTGRAY);
    for (int i = 0; i < 4; i++) {
        int ix = lx + (i % 2) * 290;
        int iy = ly + 26 + (i / 2) * 24;
        DrawRectangle(ix, iy, 16, 16, legend[i].col);
        DrawText(TextFormat("%s = %s", legend[i].key, legend[i].desc), ix + 22, iy, 17, WHITE);
    }
}

void Game::DrawPlaying() const {
    int w = GetScreenWidth();
    int h = GetScreenHeight();

    if (touchGuideAlpha > 0.01f) {
        DrawRotorTouchZone(ROTOR_FRONT_LEFT, drone, w, h, "Q", touchGuideAlpha);
        DrawRotorTouchZone(ROTOR_FRONT_RIGHT, drone, w, h, "W", touchGuideAlpha);
        DrawRotorTouchZone(ROTOR_REAR_LEFT, drone, w, h, "A", touchGuideAlpha);
        DrawRotorTouchZone(ROTOR_REAR_RIGHT, drone, w, h, "S", touchGuideAlpha);
    }

    drone.DrawHUDBars(w, h);

    // Stats (top-center, clear of the corner touch zones)
    int sx = (w - 210) / 2, sy = 16;
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

        DrawCenteredText(GetCrashTitle(crashReason), screenHeight / 2 - 100, 60, RED);
        DrawCenteredText(GetCrashHint(crashReason), screenHeight / 2 - 38, 24, LIGHTGRAY);

        float progress = fminf(drone.distanceTraveled / fabsf(PAD_WORLD_Z) * 100.0f, 100.0f);
        DrawCenteredText(TextFormat("Progress: %.0f%%", progress), screenHeight / 2, 30, WHITE);
        DrawCenteredText("Press R or tap to retry", screenHeight / 2 + 50, 24, LIGHTGRAY);
    }
}

void Game::DrawWin() const {
    DrawRectangle(0, 0, screenWidth, screenHeight, {0, 60, 0, 150});

    // Expanding celebration rings in motor colors, staggered 0.4s apart
    const Color ringColors[4] = {RED, BLUE, GREEN, YELLOW};
    const float period = 1.6f;
    int cx = screenWidth / 2, cy = screenHeight / 2;
    for (int w = 0; w < 4; w++) {
        float t      = fmod(winTimer + w * (period / 4.0f), period);
        float radius = t * 420.0f;
        float alpha  = fmaxf(0.0f, 1.0f - t / period);
        Color c      = Fade(ringColors[w], alpha * 0.85f);
        DrawCircleLines(cx, cy, radius,        c);
        DrawCircleLines(cx, cy, radius - 5.0f, Fade(ringColors[w], alpha * 0.4f));
    }

    // Pulsing title
    int titleSize = (int)(80 * (1.0f + 0.07f * sinf(winTimer * 5.0f)));
    DrawCenteredText("LANDED!",              screenHeight / 2 - 100, titleSize, GREEN);
    DrawCenteredText("Progress:  100%",      screenHeight / 2 - 10,  30, WHITE);
    DrawCenteredText("Perfect landing!",     screenHeight / 2 + 30,  24, GOLD);
    DrawCenteredText("Press R or tap to fly again", screenHeight / 2 + 70,  24, LIGHTGRAY);

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
        { "Inertia Pitch",   "kg\xb7m\xb2",&I_PITCH,     0.01f, 1.0f  },
        { "Inertia Yaw",     "kg\xb7m\xb2",&I_YAW,       0.01f, 1.0f  },
        { "Inertia Roll",    "kg\xb7m\xb2",&I_ROLL,       0.01f, 1.0f  },
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

    DrawCenteredText("Keyboard only: Up/Down Select   Left/Right Adjust   R Reset defaults   Backspace Back",
                     by + bh - 26, 16, {180, 180, 180, 255});
}
