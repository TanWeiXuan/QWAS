#include "qwas_app.h"
#include "raylib.h"

#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    const char* assistWeightsPath = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--assist-weights") == 0 && i + 1 < argc) {
            assistWeightsPath = argv[++i];
        } else {
            std::fprintf(stderr, "Usage: %s [--assist-weights path/to/model.qwasmlp]\n", argv[0]);
            return 2;
        }
    }
    QwasApp app;
    app.Init(assistWeightsPath);

    while (!WindowShouldClose()) {
        app.Frame();
    }

    app.Shutdown();
    return 0;
}
