#include "qwas_app.h"

#include "raylib.h"

namespace {
constexpr int SCREEN_WIDTH = 1280;
constexpr int SCREEN_HEIGHT = 720;
constexpr float MAX_FRAME_DT = 0.033f;
}  // namespace

void QwasApp::Init(const char* assistWeightsPath) {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "QWAS - Quadrotor With Awkward Strokes");
    SetTargetFPS(60);
    game.Init(assistWeightsPath);
}

void QwasApp::Frame() {
    float dt = GetFrameTime();
    if (dt > MAX_FRAME_DT)
        dt = MAX_FRAME_DT;

    if (!paused)
        game.Update(dt);

    game.Draw();
}

void QwasApp::Shutdown() {
    CloseWindow();
}

void QwasApp::SetPaused(bool isPaused) {
    paused = isPaused;
}
