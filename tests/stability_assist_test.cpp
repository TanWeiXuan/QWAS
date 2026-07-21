#include "drone.h"
#include "stability_assist.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
int failures = 0;

void Expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

bool Near(float actual, float expected, float tolerance) {
    return std::fabs(actual - expected) <= tolerance;
}

void TestInference(const std::filesystem::path& modelPath, const std::filesystem::path& fixturePath,
                   const std::filesystem::path& scratchPath) {
    StabilityAssistMLP baked;
    StabilityAssistMLP loaded;
    std::string error;
    Expect(loaded.LoadFromFile(modelPath.string(), &error), "portable model loads: " + error);

    std::ifstream fixture(fixturePath);
    int cases = 0;
    fixture >> cases;
    for (int testCase = 0; testCase < cases; ++testCase) {
        StabilityAssistMLP::Observation observation{};
        StabilityAssistMLP::Action expected{};
        for (float& value : observation) fixture >> value;
        for (float& value : expected) fixture >> value;
        auto bakedAction = baked.Forward(observation);
        auto loadedAction = loaded.Forward(observation);
        for (int i = 0; i < STABILITY_ASSIST_ACTION_SIZE; ++i) {
            Expect(Near(bakedAction[i], expected[i], 1.0e-5f), "baked inference parity case " + std::to_string(testCase));
            Expect(Near(loadedAction[i], expected[i], 1.0e-5f), "file inference parity case " + std::to_string(testCase));
        }
    }

    StabilityAssistMLP::Observation nonFinite{};
    nonFinite[3] = std::numeric_limits<float>::quiet_NaN();
    auto safe = loaded.Forward(nonFinite);
    for (float value : safe) Expect(value == 0.0f, "non-finite observation returns zero action");

    std::ifstream source(modelPath, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(source)), std::istreambuf_iterator<char>());
    auto tryInvalid = [&](const char* name, std::vector<char> invalid) {
        auto path = scratchPath / name;
        std::ofstream output(path, std::ios::binary);
        output.write(invalid.data(), static_cast<std::streamsize>(invalid.size()));
        output.close();
        StabilityAssistMLP candidate;
        std::string localError;
        Expect(!candidate.LoadFromFile(path.string(), &localError), std::string("rejects ") + name);
        StabilityAssistMLP::Observation zero{};
        auto fallback = candidate.Forward(zero);
        auto expectedFallback = baked.Forward(zero);
        for (int i = 0; i < STABILITY_ASSIST_ACTION_SIZE; ++i)
            Expect(Near(fallback[i], expectedFallback[i], 1.0e-7f), std::string("keeps baked fallback after ") + name);
    };
    auto invalidMagic = bytes; invalidMagic[0] ^= 0x7f; tryInvalid("invalid_magic.qwasmlp", invalidMagic);
    auto truncated = bytes; truncated.pop_back(); tryInvalid("truncated.qwasmlp", truncated);
    auto dimensions = bytes; dimensions[12] = 29; dimensions[13] = dimensions[14] = dimensions[15] = 0;
    tryInvalid("dimensions.qwasmlp", dimensions);
    auto nonFiniteWeights = bytes;
    std::uint32_t nanBits = 0x7fc00000u;
    std::memcpy(nonFiniteWeights.data() + 28 + STABILITY_ASSIST_OBSERVATION_SIZE * sizeof(float), &nanBits, sizeof(nanBits));
    tryInvalid("nonfinite.qwasmlp", nonFiniteWeights);
}

void TestPhysics(const std::filesystem::path& fixturePath) {
    std::ifstream fixture(fixturePath);
    int steps = 0;
    fixture >> steps;
    Vector3 initialPosition{}, initialVelocity{}, initialAngular{};
    Quaternion initialOrientation{};
    std::array<float, 4> initialPlayer{};
    fixture >> initialPosition.x >> initialPosition.y >> initialPosition.z;
    fixture >> initialVelocity.x >> initialVelocity.y >> initialVelocity.z;
    fixture >> initialOrientation.x >> initialOrientation.y >> initialOrientation.z >> initialOrientation.w;
    fixture >> initialAngular.x >> initialAngular.y >> initialAngular.z;
    for (float& value : initialPlayer) fixture >> value;
    fixture >> DRONE_MASS >> GRAVITY >> MAX_THRUST >> THRUST_RAMP_UP >> THRUST_RAMP_DOWN
            >> ARM_LENGTH >> I_PITCH >> I_YAW >> I_ROLL >> LIN_DRAG >> ANG_DRAG >> K_YAW;

    Drone drone;
    drone.Init(initialPosition);
    drone.velocity = initialVelocity;
    drone.orientation = initialOrientation;
    drone.angularVel = initialAngular;
    drone.playerThrust = initialPlayer;
    drone.appliedThrust = initialPlayer;
    for (int step = 0; step < steps; ++step) {
        float dt = 0.0f;
        std::array<int, 4> buttons{};
        fixture >> dt >> buttons[0] >> buttons[1] >> buttons[2] >> buttons[3];
        for (int motor = 0; motor < 4; ++motor)
            drone.SetRotorInput(static_cast<RotorID>(motor), buttons[motor] != 0, dt);
        drone.UsePlayerThrust();
        drone.Update(dt);
    }
    std::array<float, 17> expected{};
    for (float& value : expected) fixture >> value;
    std::array<float, 17> actual = {
        drone.position.x, drone.position.y, drone.position.z,
        drone.velocity.x, drone.velocity.y, drone.velocity.z,
        drone.orientation.x, drone.orientation.y, drone.orientation.z, drone.orientation.w,
        drone.angularVel.x, drone.angularVel.y, drone.angularVel.z,
        drone.playerThrust[0], drone.playerThrust[1], drone.playerThrust[2], drone.playerThrust[3],
    };
    for (std::size_t i = 0; i < actual.size(); ++i)
        Expect(Near(actual[i], expected[i], 2.0e-4f), "Python/C++ physics parity value " + std::to_string(i));
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: stability_assist_test model inference-fixture physics-fixture scratch-dir\n";
        return 2;
    }
    std::filesystem::create_directories(argv[4]);
    TestInference(argv[1], argv[2], argv[4]);
    TestPhysics(argv[3]);
    if (failures == 0)
        std::cout << "All stability assist parity and validation checks passed.\n";
    return failures == 0 ? 0 : 1;
}
