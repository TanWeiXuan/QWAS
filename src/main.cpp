#include "raylib.h"
#include "game.h"

int main() {
    const int screenWidth  = 1280;
    const int screenHeight = 720;

    InitWindow(screenWidth, screenHeight, "QWAS - Quadrotor With Awkward Strokes");
    SetTargetFPS(60);

    Game game;
    game.Init();

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        if (dt > 0.033f) dt = 0.033f;  // cap to ~30 FPS physics on frame spikes
        game.Update(dt);
        game.Draw();
    }

    CloseWindow();
    return 0;
}
