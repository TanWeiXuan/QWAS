#pragma once

#include "game.h"

class QwasApp {
public:
    void Init(const char* assistWeightsPath = nullptr);
    void Frame();
    void Shutdown();
    void SetPaused(bool paused);

private:
    Game game = {};
    bool paused = false;
};
