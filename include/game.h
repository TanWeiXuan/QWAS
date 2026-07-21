#pragma once
#include "raylib.h"
#include "drone.h"
#include "stability_assist.h"

constexpr float CHASE_DIST    = 4.0f;
constexpr float CHASE_HEIGHT  = 2.0f;
constexpr float CAMERA_FOV    = 60.0f;
constexpr float CAMERA_SMOOTH = 3.0f;

constexpr float PAD_WORLD_Z   = -25.0f;  // landing pad distance (meters forward)
constexpr float PAD_TOP_Y    = 0.10f;   // top surface of any pad (cube: center 0.05, height 0.10)
constexpr float DRONE_REST_Y = 0.13f;   // drone center when resting on a pad (PAD_TOP_Y + body half-height 0.03)

enum class GameState { MENU, SETTINGS, INSTRUCTIONS, PLAYING, DEAD, WIN };
enum class CrashReason { NONE, ROTOR_STRIKE, GROUND_IMPACT, TOO_HIGH, OUT_OF_BOUNDS };

struct LandingPad {
    Vector3 position;  // center
    float   halfSize;  // half-width of square pad
};

struct Game {
    int        screenWidth;
    int        screenHeight;
    GameState  state;
    CrashReason crashReason;
    Drone      drone;
    LandingPad startPad;  // spawn pad; ground contact here is never a crash
    LandingPad pad;       // destination pad
    Camera3D   camera;
    float      deadTimer;        // counts down; press R only when <= 0
    float      winTimer;         // counts up from 0 on WIN entry (drives celebration)
    float      bestScore;        // best progress % this session
    int        settingsSelectedIdx;
    bool       draggingSlider;
    int        draggedSettingsIdx;
    int        menuSelectedIdx;
    bool       easyMode;
    StabilityAssistMLP stabilityAssist;
    StabilityAssistController stabilityController;
    int        endScreenSelectedIdx;  // 0=Retry, 1=Menu on DEAD/WIN screens
    float      touchGuideAlpha;  // fades after the first motor input
    bool       touchGuideDismissed;
    bool       tapWasDown;
    bool       tapCandidate;
    Vector2    tapStart;

    void Init(const char* assistWeightsPath = nullptr);
    void Reset();
    void Update(float dt);
    void Draw() const;

private:
    void UpdateMenu();
    void UpdateSettings();
    void UpdateInstructions();
    void UpdatePlaying(float dt);
    void UpdateDead(float dt);
    void UpdateWin(float dt);

    void DrawMenu() const;
    void DrawSettings() const;
    void DrawInstructions() const;
    void DrawPlaying() const;
    void DrawDead() const;
    void DrawWin() const;
    void ActivateMenuButton(int idx);

    void UpdateCamera(float dt);
    void CheckGameStatus();
    void DrawWorld() const;
    void DrawOverlay() const;
    bool ConsumeCompletedTap();
};
