#include "game.h"
#include "raymath.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

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
    bool active = drone.appliedThrust[id] > 0.01f;

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
static Rectangle GetEndScreenButtonRect(int idx, int screenW, int btnTopY) {
    const int btnW = 280, btnH = 52, gap = 24;
    int bx = (screenW - (btnW * 2 + gap)) / 2 + idx * (btnW + gap);
    return {(float)bx, (float)btnTopY, (float)btnW, (float)btnH};
}

static Rectangle GetMenuButtonRect(int idx, int screenW, int screenH) {
    const int btnW = 400, btnH = 52;
    int btnX = (screenW - btnW) / 2;
    int btnY = (int)(screenH * 0.43f) + idx * 66;
    return {(float)btnX, (float)btnY, (float)btnW, (float)btnH};
}

static void DrawMenuButton(const char* label, Rectangle rect, bool selected) {
    Color bg     = selected ? Color{45, 75, 45, 230} : Color{15, 15, 15, 210};
    Color border = selected ? Color{100, 200, 100, 255} : Color{55, 55, 55, 200};
    DrawRectangleRec(rect, bg);
    DrawRectangleLinesEx(rect, 2.0f, border);
    const int fs = 26;
    int tw = MeasureText(label, fs);
    DrawText(label,
             (int)(rect.x + (rect.width  - tw) * 0.5f),
             (int)(rect.y + (rect.height - fs) * 0.5f),
             fs, selected ? WHITE : Color{200, 200, 200, 255});
}

struct SettingsEntry {
    const char* label;
    const char* unit;
    float*      val;
    float       minV;
    float       maxV;
    float       step;
    float       defaultVal;
};

constexpr int SETTINGS_COUNT = 12;
constexpr int SETTINGS_ROW_H = 34;

const SettingsEntry kSettingsEntries[SETTINGS_COUNT] = {
    { "Mass",             "kg",          &DRONE_MASS,       0.1f,  5.0f,  0.05f, DRONE_MASS_DEFAULT       },
    { "Gravity",          "m/s\xb2",     &GRAVITY,          0.0f,  20.0f, 0.1f,  GRAVITY_DEFAULT          },
    { "Max Thrust",       "N",           &MAX_THRUST,       0.5f,  10.0f, 0.1f,  MAX_THRUST_DEFAULT       },
    { "Thrust Ramp Up",   "N/s",         &THRUST_RAMP_UP,   1.0f,  20.0f, 0.5f,  THRUST_RAMP_UP_DEFAULT   },
    { "Thrust Ramp Down", "N/s",         &THRUST_RAMP_DOWN, 1.0f,  30.0f, 0.5f,  THRUST_RAMP_DOWN_DEFAULT },
    { "Arm Length",       "m",           &ARM_LENGTH,       0.05f, 1.0f,  0.01f, ARM_LENGTH_DEFAULT       },
    { "Inertia Pitch",    "kg\xb7m\xb2", &I_PITCH,          0.01f, 1.0f,  0.05f, I_PITCH_DEFAULT          },
    { "Inertia Yaw",      "kg\xb7m\xb2", &I_YAW,            0.01f, 1.0f,  0.05f, I_YAW_DEFAULT            },
    { "Inertia Roll",     "kg\xb7m\xb2", &I_ROLL,           0.01f, 1.0f,  0.05f, I_ROLL_DEFAULT           },
    { "Linear Drag",      "/s",          &LIN_DRAG,         0.0f,  3.0f,  0.5f,  LIN_DRAG_DEFAULT         },
    { "Angular Drag",     "/s",          &ANG_DRAG,         0.0f,  10.0f, 0.1f,  ANG_DRAG_DEFAULT         },
    { "Yaw Coeff",        "",            &K_YAW,            0.01f, 0.2f,  0.01f, K_YAW_DEFAULT            },
};

// Shared modal-dialog chrome: full-screen dim + panel fill/border, and a
// BACK button + footer hint reserved in a fixed-height block at the bottom
// of every panel, so Settings/Instructions/any future dialog look and
// behave identically.
constexpr int DIALOG_BACK_BTN_W            = 220;
constexpr int DIALOG_BACK_BTN_H            = 52;
constexpr int DIALOG_BACK_BTN_BOTTOM_MARGIN = 20;
constexpr int DIALOG_FOOTER_GAP            = 24;  // gap between footer hint and button top
constexpr int DIALOG_FOOTER_BLOCK_H        = DIALOG_BACK_BTN_BOTTOM_MARGIN + DIALOG_BACK_BTN_H + DIALOG_FOOTER_GAP;
constexpr int DIALOG_TITLE_SIZE            = 28;
constexpr int DIALOG_TITLE_TOP_PAD         = 14;

Rectangle GetDialogBackButtonRect(int bx, int by, int bw, int bh) {
    int bx2 = bx + (bw - DIALOG_BACK_BTN_W) / 2;
    int by2 = by + bh - DIALOG_BACK_BTN_BOTTOM_MARGIN - DIALOG_BACK_BTN_H;
    return { (float)bx2, (float)by2, (float)DIALOG_BACK_BTN_W, (float)DIALOG_BACK_BTN_H };
}

int GetDialogFooterY(int by, int bh) {
    return by + bh - DIALOG_FOOTER_BLOCK_H;
}

struct SettingsLayout { int bx, by, bw, bh, sliderX, sliderW; };

SettingsLayout GetSettingsLayout(int screenW, int screenH) {
    int bw = 820, bh = 60 + SETTINGS_COUNT * SETTINGS_ROW_H + DIALOG_FOOTER_BLOCK_H + 20;
    int bx = (screenW - bw) / 2;
    int by = (screenH - bh) / 2;
    return { bx, by, bw, bh, bx + 290, 300 };
}

Rectangle GetSettingsRowRect(const SettingsLayout& L, int idx) {
    int ry = L.by + 54 + idx * SETTINGS_ROW_H;
    return { (float)(L.bx + 2), (float)ry, (float)(L.bw - 4), (float)(SETTINGS_ROW_H - 2) };
}

Rectangle GetSettingsSliderHitRect(const SettingsLayout& L, int idx) {
    int ry = L.by + 54 + idx * SETTINGS_ROW_H;
    return { (float)L.sliderX, (float)ry, (float)L.sliderW, (float)SETTINGS_ROW_H };
}

struct InstructionsLayout { int bx, by, bw, bh; };

InstructionsLayout GetInstructionsLayout(int screenW, int screenH) {
    const int bw = 760, bh = 680;
    int bx = (screenW - bw) / 2;
    int by = (screenH - bh) / 2;
    return { bx, by, bw, bh };
}
}  // namespace

// ---------------------------------------------------------------------------
//  Lifecycle
// ---------------------------------------------------------------------------

void Game::Init(const char* assistWeightsPath) {
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
    draggingSlider = false;
    draggedSettingsIdx = 0;
    menuSelectedIdx = 0;
    easyMode = false;
    endScreenSelectedIdx = 0;
    touchGuideAlpha = 1.0f;
    touchGuideDismissed = false;
    tapWasDown = false;
    tapCandidate = false;
    tapStart = {0, 0};
    stabilityController.Reset();

    if (assistWeightsPath) {
        std::string error;
        if (!stabilityAssist.LoadFromFile(assistWeightsPath, &error))
            std::fprintf(stderr, "Warning: could not load Easy Mode weights '%s': %s; using baked weights.\n",
                         assistWeightsPath, error.c_str());
    }
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
    stabilityController.Reset();

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
        case GameState::MENU:           UpdateMenu();         break;
        case GameState::SETTINGS:       UpdateSettings();     break;
        case GameState::INSTRUCTIONS:   UpdateInstructions(); break;
        case GameState::PLAYING:        UpdatePlaying(dt);      break;
        case GameState::DEAD:           UpdateDead(dt);         break;
        case GameState::WIN:            UpdateWin(dt);          break;
    }
}

void Game::UpdateMenu() {
    constexpr int MENU_COUNT = 4;
    int sw = GetScreenWidth(), sh = GetScreenHeight();

    // Keyboard navigation
    if (IsKeyPressed(KEY_UP))   menuSelectedIdx = (menuSelectedIdx - 1 + MENU_COUNT) % MENU_COUNT;
    if (IsKeyPressed(KEY_DOWN)) menuSelectedIdx = (menuSelectedIdx + 1) % MENU_COUNT;

    // Mouse hover → update selection
    Vector2 mp = GetMousePosition();
    for (int i = 0; i < MENU_COUNT; i++) {
        if (CheckCollisionPointRec(mp, GetMenuButtonRect(i, sw, sh)))
            menuSelectedIdx = i;
    }
    // Touch hover → update selection
    for (int t = 0; t < GetTouchPointCount(); t++) {
        Vector2 tp = GetTouchPosition(t);
        for (int i = 0; i < MENU_COUNT; i++) {
            if (CheckCollisionPointRec(tp, GetMenuButtonRect(i, sw, sh)))
                menuSelectedIdx = i;
        }
    }

    // Activate via keyboard Enter / Space
    if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_SPACE)) {
        ActivateMenuButton(menuSelectedIdx);
        return;
    }

    // Activate via mouse click
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        for (int i = 0; i < MENU_COUNT; i++) {
            if (CheckCollisionPointRec(mp, GetMenuButtonRect(i, sw, sh))) {
                ActivateMenuButton(i);
                return;
            }
        }
    }

    // Activate via touch tap
    if (ConsumeCompletedTap()) {
        for (int i = 0; i < MENU_COUNT; i++) {
            if (CheckCollisionPointRec(tapStart, GetMenuButtonRect(i, sw, sh))) {
                ActivateMenuButton(i);
                return;
            }
        }
    }
}

void Game::ActivateMenuButton(int idx) {
    switch (idx) {
        case 0: Reset(); endScreenSelectedIdx = 0; state = GameState::PLAYING; break;
        case 1: easyMode = !easyMode; break;
        case 2: settingsSelectedIdx = 0; draggingSlider = false; state = GameState::SETTINGS; break;
        case 3: state = GameState::INSTRUCTIONS; break;
    }
}

void Game::UpdateSettings() {
    auto exitToMenu = [&] {
        draggingSlider = false;
        menuSelectedIdx = 2;
        state = GameState::MENU;
    };

    if (IsKeyPressed(KEY_BACKSPACE) || IsKeyPressed(KEY_ESCAPE)) { exitToMenu(); return; }

    if (IsKeyPressed(KEY_R)) {
        for (const SettingsEntry& e : kSettingsEntries) *e.val = e.defaultVal;
        return;
    }

    if (IsKeyPressed(KEY_UP))   settingsSelectedIdx = (settingsSelectedIdx - 1 + SETTINGS_COUNT) % SETTINGS_COUNT;
    if (IsKeyPressed(KEY_DOWN)) settingsSelectedIdx = (settingsSelectedIdx + 1) % SETTINGS_COUNT;

    const SettingsEntry& ke = kSettingsEntries[settingsSelectedIdx];
    if (IsKeyDown(KEY_LEFT))  *ke.val = fmaxf(ke.minV, *ke.val - ke.step);
    if (IsKeyDown(KEY_RIGHT)) *ke.val = fminf(ke.maxV, *ke.val + ke.step);

    // Pointer (mouse/touch) handling for sliders
    SettingsLayout L = GetSettingsLayout(screenWidth, screenHeight);
    bool pointerDown = IsPrimaryPointerDown();

    if (draggingSlider) {
        if (pointerDown) {
            float px = GetPrimaryPointerPosition().x;
            const SettingsEntry& e = kSettingsEntries[draggedSettingsIdx];
            float t = Clamp((px - L.sliderX) / (float)L.sliderW, 0.0f, 1.0f);
            *e.val = e.minV + t * (e.maxV - e.minV);
        } else {
            draggingSlider = false;
        }
    } else if (pointerDown) {
        Vector2 pp = GetPrimaryPointerPosition();
        for (int i = 0; i < SETTINGS_COUNT; i++) {
            if (CheckCollisionPointRec(pp, GetSettingsRowRect(L, i))) {
                settingsSelectedIdx = i;
                if (CheckCollisionPointRec(pp, GetSettingsSliderHitRect(L, i))) {
                    draggingSlider = true;
                    draggedSettingsIdx = i;
                    const SettingsEntry& e = kSettingsEntries[i];
                    float t = Clamp((pp.x - L.sliderX) / (float)L.sliderW, 0.0f, 1.0f);
                    *e.val = e.minV + t * (e.maxV - e.minV);
                }
                break;
            }
        }
    }

    // Back button (discrete — skip while a slider drag is in progress)
    if (!draggingSlider) {
        Rectangle backRect = GetDialogBackButtonRect(L.bx, L.by, L.bw, L.bh);
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && CheckCollisionPointRec(GetMousePosition(), backRect)) {
            exitToMenu(); return;
        }
        if (ConsumeCompletedTap() && CheckCollisionPointRec(tapStart, backRect)) {
            exitToMenu(); return;
        }
    }
}

void Game::UpdateInstructions() {
    if (IsKeyPressed(KEY_BACKSPACE) || IsKeyPressed(KEY_ESCAPE)) {
        menuSelectedIdx = 3;
        state = GameState::MENU;
        return;
    }

    InstructionsLayout L = GetInstructionsLayout(GetScreenWidth(), GetScreenHeight());
    Rectangle backRect = GetDialogBackButtonRect(L.bx, L.by, L.bw, L.bh);

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && CheckCollisionPointRec(GetMousePosition(), backRect)) {
        menuSelectedIdx = 3;
        state = GameState::MENU;
        return;
    }
    if (ConsumeCompletedTap() && CheckCollisionPointRec(tapStart, backRect)) {
        menuSelectedIdx = 3;
        state = GameState::MENU;
    }
}

void Game::UpdatePlaying(float dt) {
    if (IsKeyPressed(KEY_BACKSPACE) || IsKeyPressed(KEY_ESCAPE)) { state = GameState::MENU; return; }
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

    std::array<bool, ROTOR_COUNT> playerButtons = {
        frontLeftInput, frontRightInput, rearLeftInput, rearRightInput};
    if (easyMode)
        stabilityController.Apply(drone, playerButtons, dt, stabilityAssist);
    else
        drone.UsePlayerThrust();

    drone.Update(dt);
    UpdateCamera(dt);
    CheckGameStatus();

    float progress = fminf(drone.distanceTraveled / fabsf(PAD_WORLD_Z) * 100.0f, 100.0f);
    if (progress > bestScore) bestScore = progress;
}

void Game::UpdateDead(float dt) {
    deadTimer -= dt;

    if (IsKeyPressed(KEY_BACKSPACE) || IsKeyPressed(KEY_ESCAPE)) { state = GameState::MENU; return; }

    if (deadTimer > 0.0f) return;

    int sw = GetScreenWidth(), sh = GetScreenHeight();
    int btnY = sh / 2 + 90;

    // Keyboard navigation between Retry (0) and Menu (1)
    if (IsKeyPressed(KEY_LEFT))  endScreenSelectedIdx = 0;
    if (IsKeyPressed(KEY_RIGHT)) endScreenSelectedIdx = 1;

    // Hover: mouse
    Vector2 mp = GetMousePosition();
    for (int i = 0; i < 2; i++)
        if (CheckCollisionPointRec(mp, GetEndScreenButtonRect(i, sw, btnY)))
            endScreenSelectedIdx = i;
    // Hover: touch
    for (int t = 0; t < GetTouchPointCount(); t++) {
        Vector2 tp = GetTouchPosition(t);
        for (int i = 0; i < 2; i++)
            if (CheckCollisionPointRec(tp, GetEndScreenButtonRect(i, sw, btnY)))
                endScreenSelectedIdx = i;
    }

    auto activateEnd = [&](int idx) {
        if (idx == 0) { Reset(); state = GameState::PLAYING; }
        else          { state = GameState::MENU; }
    };

    if (IsKeyPressed(KEY_R))                                    { activateEnd(0); return; }
    if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_SPACE))     { activateEnd(endScreenSelectedIdx); return; }

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        for (int i = 0; i < 2; i++)
            if (CheckCollisionPointRec(mp, GetEndScreenButtonRect(i, sw, btnY))) { activateEnd(i); return; }
    }
    if (ConsumeCompletedTap()) {
        for (int i = 0; i < 2; i++)
            if (CheckCollisionPointRec(tapStart, GetEndScreenButtonRect(i, sw, btnY))) { activateEnd(i); return; }
    }
}

void Game::UpdateWin(float dt) {
    winTimer += dt;

    if (IsKeyPressed(KEY_BACKSPACE) || IsKeyPressed(KEY_ESCAPE)) { state = GameState::MENU; return; }

    int sw = GetScreenWidth(), sh = GetScreenHeight();
    int btnY = sh / 2 + 90;

    // Keyboard navigation between Retry (0) and Menu (1)
    if (IsKeyPressed(KEY_LEFT))  endScreenSelectedIdx = 0;
    if (IsKeyPressed(KEY_RIGHT)) endScreenSelectedIdx = 1;

    // Hover: mouse
    Vector2 mp = GetMousePosition();
    for (int i = 0; i < 2; i++)
        if (CheckCollisionPointRec(mp, GetEndScreenButtonRect(i, sw, btnY)))
            endScreenSelectedIdx = i;
    // Hover: touch
    for (int t = 0; t < GetTouchPointCount(); t++) {
        Vector2 tp = GetTouchPosition(t);
        for (int i = 0; i < 2; i++)
            if (CheckCollisionPointRec(tp, GetEndScreenButtonRect(i, sw, btnY)))
                endScreenSelectedIdx = i;
    }

    auto activateEnd = [&](int idx) {
        if (idx == 0) { Reset(); state = GameState::PLAYING; }
        else          { state = GameState::MENU; }
    };

    if (IsKeyPressed(KEY_R))                                    { activateEnd(0); return; }
    if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_SPACE))     { activateEnd(endScreenSelectedIdx); return; }

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        for (int i = 0; i < 2; i++)
            if (CheckCollisionPointRec(mp, GetEndScreenButtonRect(i, sw, btnY))) { activateEnd(i); return; }
    }
    if (ConsumeCompletedTap()) {
        for (int i = 0; i < 2; i++)
            if (CheckCollisionPointRec(tapStart, GetEndScreenButtonRect(i, sw, btnY))) { activateEnd(i); return; }
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

void Game::CheckGameStatus() {
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
    bool reachedPad = fabsf(drone.position.x - pad.position.x) < pad.halfSize &&
                   fabsf(drone.position.z - pad.position.z) < pad.halfSize &&
                   drone.position.y < DRONE_REST_Y + 0.5f;
    if (reachedPad) {
        state     = GameState::WIN;
        bestScore = 99.999f;

        // Perfect win - only if drone lands gently and level
        constexpr float perfectLandSpeed = 2.0f;
        constexpr float perfectLandAngle = 20.0f;
        float speed = Vector3Length(drone.velocity);
        bool perfectWin = speed < perfectLandSpeed && drone.GetTiltAngle() < perfectLandAngle;
        if (perfectWin) {
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
        case GameState::SETTINGS:      DrawSettings();      break;
        case GameState::INSTRUCTIONS:  DrawInstructions();  break;
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

// Shared chrome for boxed modal dialogs (Settings, Instructions): dimmed
// backdrop, filled/bordered panel, and title, so both screens look alike.
static void DrawDialogPanel(int bx, int by, int bw, int bh, int screenW, int screenH, const char* title) {
    DrawRectangle(0, 0, screenW, screenH, {0, 0, 0, 180});
    DrawRectangle(bx, by, bw, bh, {10, 10, 10, 220});
    DrawRectangleLinesEx({(float)bx, (float)by, (float)bw, (float)bh}, 2.0f, {70, 70, 70, 255});
    DrawCenteredText(title, by + DIALOG_TITLE_TOP_PAD, DIALOG_TITLE_SIZE, WHITE);
}

// Shared footer for boxed modal dialogs: hint text + BACK button, both
// positioned in the fixed-height block reserved by DIALOG_FOOTER_BLOCK_H.
static void DrawDialogFooter(int bx, int by, int bw, int bh, const char* hint) {
    DrawCenteredText(hint, GetDialogFooterY(by, bh), 16, {180, 180, 180, 255});
    Rectangle backRect = GetDialogBackButtonRect(bx, by, bw, bh);
    bool backHovered = CheckCollisionPointRec(GetMousePosition(), backRect);
    DrawMenuButton("BACK", backRect, backHovered);
}

void Game::DrawMenu() const {
    int sw = GetScreenWidth(), sh = GetScreenHeight();

    // Subtle darkening so world is still visible as backdrop
    DrawRectangle(0, 0, sw, sh, {0, 0, 0, 110});

    // Title with drop shadow
    const char* title = "QWAS";
    const int titleSize = 104;
    int tw = MeasureText(title, titleSize);
    int tx = (sw - tw) / 2;
    int ty = (int)(sh * 0.12f);
    DrawText(title, tx + 4, ty + 4, titleSize, {0, 0, 0, 150});
    DrawText(title, tx, ty, titleSize, WHITE);

    // Subtitle
    const char* sub = "Quadrotor With Awkward Strokes";
    const int subSize = 32;
    int stw = MeasureText(sub, subSize);
    DrawText(sub, (sw - stw) / 2, ty + titleSize + 12, subSize, {180, 180, 180, 255});

    // Menu buttons
    const char* labels[] = {"START", easyMode ? "EASY MODE: ON" : "EASY MODE: OFF", "SETTINGS", "INSTRUCTIONS"};
    for (int i = 0; i < 4; i++)
        DrawMenuButton(labels[i], GetMenuButtonRect(i, sw, sh), i == menuSelectedIdx);

    DrawCenteredText("Arrow keys to navigate  |  Enter or click to select",
                     sh - 28, 14, {100, 100, 100, 255});
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

    if (easyMode)
        DrawText("EASY MODE", 16, 16, 18, Fade(LIME, 0.85f));

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
    int sw = GetScreenWidth(), sh = GetScreenHeight();
    drone.DrawHUDBars(sw, sh);

    if (deadTimer <= 0.0f) {
        DrawRectangle(0, 0, sw, sh, {0, 0, 0, 120});

        DrawCenteredText(GetCrashTitle(crashReason), sh / 2 - 100, 60, RED);
        DrawCenteredText(GetCrashHint(crashReason),  sh / 2 - 38,  24, LIGHTGRAY);

        float progress = fminf(drone.distanceTraveled / fabsf(PAD_WORLD_Z) * 100.0f, 100.0f);
        DrawCenteredText(TextFormat("Progress: %.0f%%", progress), sh / 2, 30, WHITE);

        int btnY = sh / 2 + 90;
        DrawMenuButton("RETRY",           GetEndScreenButtonRect(0, sw, btnY), endScreenSelectedIdx == 0);
        DrawMenuButton("RETURN TO MENU",  GetEndScreenButtonRect(1, sw, btnY), endScreenSelectedIdx == 1);
        DrawCenteredText("Left/Right to navigate  |  Enter or click to select",
                         btnY + 66, 14, {100, 100, 100, 255});
    }
}

void Game::DrawWin() const {
    int sw = GetScreenWidth(), sh = GetScreenHeight();
    DrawRectangle(0, 0, sw, sh, {0, 60, 0, 150});

    // Expanding celebration rings in motor colors, staggered 0.4s apart
    const Color ringColors[4] = {RED, BLUE, GREEN, YELLOW};
    const float period = 1.6f;
    int cx = sw / 2, cy = sh / 2;
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
    if (bestScore < 100.0f)
    {
        DrawCenteredText("\"LANDED\"",              sh / 2 - 100, titleSize, GREEN);
        DrawCenteredText("Progress:  99.999%",  sh / 2 - 10,  30, WHITE);
        DrawCenteredText("You made it to the end.",    sh / 2 + 30,  24, GOLD);
    }
    else
    {
        DrawCenteredText("LANDED",              sh / 2 - 100, titleSize, GREEN);
        DrawCenteredText("Progress:  100%",  sh / 2 - 10,  30, WHITE);
        DrawCenteredText("Perfect landing!",    sh / 2 + 30,  24, GOLD);
    }

    int btnY = sh / 2 + 90;
    DrawMenuButton("FLY AGAIN",       GetEndScreenButtonRect(0, sw, btnY), endScreenSelectedIdx == 0);
    DrawMenuButton("RETURN TO MENU",  GetEndScreenButtonRect(1, sw, btnY), endScreenSelectedIdx == 1);
    DrawCenteredText("Left/Right to navigate  |  Enter or click to select",
                     btnY + 66, 14, {100, 100, 100, 255});

    drone.DrawHUDBars(sw, sh);
}

void Game::DrawSettings() const {
    SettingsLayout L = GetSettingsLayout(screenWidth, screenHeight);
    DrawDialogPanel(L.bx, L.by, L.bw, L.bh, screenWidth, screenHeight, "PHYSICS SETTINGS");

    int labelX = L.bx + 20;
    int valueX = L.bx + L.bw - 120;

    for (int i = 0; i < SETTINGS_COUNT; i++) {
        int ry = L.by + 54 + i * SETTINGS_ROW_H;
        const SettingsEntry& e = kSettingsEntries[i];

        if (i == settingsSelectedIdx)
            DrawRectangle(L.bx + 2, ry, L.bw - 4, SETTINGS_ROW_H - 2, {255, 255, 255, 30});

        Color textCol = (i == settingsSelectedIdx) ? WHITE : LIGHTGRAY;

        // Label
        DrawText(e.label, labelX, ry + 8, 18, textCol);

        // Slider track
        DrawRectangle(L.sliderX, ry + 12, L.sliderW, 10, {80, 80, 80, 255});

        // Slider fill
        float t = (*e.val - e.minV) / (e.maxV - e.minV);
        int fillW = (int)(t * L.sliderW);
        Color fillCol = (i == settingsSelectedIdx) ? WHITE : GRAY;
        DrawRectangle(L.sliderX, ry + 12, fillW, 10, fillCol);

        // Slider thumb
        DrawRectangle(L.sliderX + fillW - 3, ry + 8, 6, 18,
                      (i == settingsSelectedIdx) ? YELLOW : DARKGRAY);

        // Value + unit
        DrawText(TextFormat("%.3g %s", *e.val, e.unit), valueX, ry + 8, 17, textCol);
    }

    DrawDialogFooter(L.bx, L.by, L.bw, L.bh,
                     "Drag sliders or Up/Down + Left/Right to adjust   R Reset defaults   Back or Backspace/Escape to exit");
}

void Game::DrawInstructions() const {
    int sw = GetScreenWidth(), sh = GetScreenHeight();
    InstructionsLayout L = GetInstructionsLayout(sw, sh);
    const int bx = L.bx, by = L.by, bw = L.bw, bh = L.bh;
    DrawDialogPanel(bx, by, bw, bh, sw, sh, "INSTRUCTIONS");

    int lx = bx + 40;
    int y  = by + 65;

    DrawText("Objective", lx, y, 20, YELLOW);
    y += 26;
    DrawText("Fly from the green start pad to the orange landing pad.", lx, y, 17, WHITE);
    y += 20;
    DrawText("Land gently (low speed, nearly level) to win.", lx, y, 17, WHITE);
    y += 34;

    DrawText("Keyboard Controls", lx, y, 20, YELLOW);
    y += 26;
    struct MotorKey { const char* key; Color col; const char* desc; };
    const MotorKey motors[4] = {
        {"Q", RED,    "Front-Left rotor"},
        {"W", BLUE,   "Front-Right rotor"},
        {"A", GREEN,  "Rear-Left rotor"},
        {"S", YELLOW, "Rear-Right rotor"},
    };
    for (auto& m : motors) {
        DrawRectangle(lx, y + 2, 14, 14, m.col);
        DrawText(m.key,  lx + 18, y, 17, WHITE);
        DrawText(m.desc, lx + 38, y, 17, LIGHTGRAY);
        y += 22;
    }
    y += 4;
    DrawText("R  -  Restart", lx, y, 17, WHITE);
    y += 34;

    DrawText("Touch / Click Controls", lx, y, 20, YELLOW);
    y += 26;
    DrawText("Hold a screen corner to spin the matching rotor:", lx, y, 17, WHITE);
    y += 20;
    DrawText("Top-Left = Q          Top-Right = W", lx, y, 17, WHITE);
    y += 20;
    DrawText("Bottom-Left = A       Bottom-Right = S", lx, y, 17, WHITE);
    y += 34;

    DrawText("Flight Tips", lx, y, 20, YELLOW);
    y += 26;
    DrawText("Longer hold = more thrust.  Release = thrust drops quickly.", lx, y, 17, WHITE);
    y += 20;
    DrawText("The drone tilts toward whichever rotors are spinning harder.", lx, y, 17, WHITE);
    y += 20;
    DrawText("Use Settings to adjust physics constants.", lx, y, 17, WHITE);

    DrawDialogFooter(bx, by, bw, bh, "Back or Backspace/Escape to exit");
}
