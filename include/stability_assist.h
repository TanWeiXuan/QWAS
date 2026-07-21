#pragma once

#include <array>
#include <string>

struct Drone;

constexpr int STABILITY_ASSIST_OBSERVATION_SIZE = 32;
constexpr int STABILITY_ASSIST_HIDDEN_SIZE = 32;
constexpr int STABILITY_ASSIST_ACTION_SIZE = 2;

inline constexpr float EASY_MODE_SINK_THRUST_RATIO = 0.95f;
inline constexpr float EASY_MODE_RELEASE_BLEND_START = 0.04f;
inline constexpr float EASY_MODE_RELEASE_BLEND_FULL = 0.20f;
inline constexpr float EASY_MODE_RELEASE_DIFFERENTIAL_TIME_CONSTANT = 0.14f;
inline constexpr float EASY_MODE_ACTIVE_DIFFERENTIAL_THRUST_RATIO = 0.30f;
inline constexpr float EASY_MODE_RELEASE_DIFFERENTIAL_THRUST_RATIO = 0.50f;
inline constexpr float EASY_MODE_RELEASE_PITCH_KP = 1.80f;
inline constexpr float EASY_MODE_RELEASE_PITCH_KD = 0.70f;
inline constexpr float EASY_MODE_RELEASE_ROLL_KP = 1.80f;
inline constexpr float EASY_MODE_RELEASE_ROLL_KD = 0.70f;
inline constexpr float EASY_MODE_RELEASE_RESIDUAL_SCALE = 0.20f;

// Canonical deployment observation order. tools/stability_assist/spec.py is
// mechanically checked against these values by the Python test suite.
enum StabilityAssistObservationIndex {
    ASSIST_BODY_UP_X = 0,
    ASSIST_BODY_UP_Y = 1,
    ASSIST_BODY_UP_Z = 2,
    ASSIST_ANGULAR_VELOCITY_X = 3,
    ASSIST_ANGULAR_VELOCITY_Y = 4,
    ASSIST_ANGULAR_VELOCITY_Z = 5,
    ASSIST_PLAYER_THRUST_FRONT_LEFT = 6,
    ASSIST_PLAYER_THRUST_FRONT_RIGHT = 7,
    ASSIST_PLAYER_THRUST_REAR_LEFT = 8,
    ASSIST_PLAYER_THRUST_REAR_RIGHT = 9,
    ASSIST_BUTTON_FRONT_LEFT = 10,
    ASSIST_BUTTON_FRONT_RIGHT = 11,
    ASSIST_BUTTON_REAR_LEFT = 12,
    ASSIST_BUTTON_REAR_RIGHT = 13,
    ASSIST_FRAME_DT = 14,
    ASSIST_INTERVENTION_GATE = 15,
    ASSIST_MASS = 16,
    ASSIST_GRAVITY = 17,
    ASSIST_MAXIMUM_THRUST = 18,
    ASSIST_THRUST_RAMP_UP = 19,
    ASSIST_THRUST_RAMP_DOWN = 20,
    ASSIST_ARM_LENGTH = 21,
    ASSIST_PITCH_INERTIA = 22,
    ASSIST_YAW_INERTIA = 23,
    ASSIST_ROLL_INERTIA = 24,
    ASSIST_LINEAR_DRAG = 25,
    ASSIST_ANGULAR_DRAG = 26,
    ASSIST_YAW_COEFFICIENT = 27,
    ASSIST_RELEASE_BLEND = 28,
    ASSIST_TIME_SINCE_PLAYER_INPUT = 29,
    ASSIST_PREVIOUS_PITCH_RESIDUAL = 30,
    ASSIST_PREVIOUS_ROLL_RESIDUAL = 31,
};

// Raw observation order is documented in doc/stability_assist.md. Forward()
// applies the model's fixed normalization scales and clamps to [-4, 4].
class StabilityAssistMLP {
public:
    using Observation = std::array<float, STABILITY_ASSIST_OBSERVATION_SIZE>;
    using Action = std::array<float, STABILITY_ASSIST_ACTION_SIZE>;

    StabilityAssistMLP();

    Action Forward(const Observation& rawObservation) const;
    bool LoadFromFile(const std::string& path, std::string* error = nullptr);
    void UseBakedWeights();

private:
    std::array<float, STABILITY_ASSIST_OBSERVATION_SIZE> observationScale_{};
    std::array<float, STABILITY_ASSIST_HIDDEN_SIZE * STABILITY_ASSIST_OBSERVATION_SIZE> weights1_{};
    std::array<float, STABILITY_ASSIST_HIDDEN_SIZE> bias1_{};
    std::array<float, STABILITY_ASSIST_HIDDEN_SIZE * STABILITY_ASSIST_HIDDEN_SIZE> weights2_{};
    std::array<float, STABILITY_ASSIST_HIDDEN_SIZE> bias2_{};
    std::array<float, STABILITY_ASSIST_ACTION_SIZE * STABILITY_ASSIST_HIDDEN_SIZE> weights3_{};
    std::array<float, STABILITY_ASSIST_ACTION_SIZE> bias3_{};
};

struct StabilityAssistTelemetry {
    float releaseBlend = 0.0f;
    float dangerGate = 0.0f;
    float releaseRecoveryGate = 0.0f;
    float effectiveGate = 0.0f;
    float pitchPD = 0.0f;
    float rollPD = 0.0f;
    float pitchResidual = 0.0f;
    float rollResidual = 0.0f;
    float pitchCommand = 0.0f;
    float rollCommand = 0.0f;
    float assistMagnitude = 0.0f;
    bool pdSaturated = false;
};

class StabilityAssistController {
public:
    void Reset();
    StabilityAssistTelemetry Apply(
        Drone& drone,
        const std::array<bool, 4>& playerButtons,
        float dt,
        const StabilityAssistMLP& model);

    float TimeSincePlayerInput() const { return timeSincePlayerInput_; }
    float ReleaseBlend() const { return releaseBlend_; }
    std::array<float, 2> PreviousResidual() const {
        return {previousPitchResidual_, previousRollResidual_};
    }

private:
    float timeSincePlayerInput_ = 0.0f;
    float releaseBlend_ = 0.0f;
    float previousPitchResidual_ = 0.0f;
    float previousRollResidual_ = 0.0f;
};

float ComputeStabilityDangerGate(float tiltRadians, float pitchRollRate);
float ComputeStabilityReleaseRecoveryGate(float tiltRadians, float pitchRollRate);
float ComputeStabilityReleaseBlend(float timeSincePlayerInput);
std::array<float, 2> ComputeStabilityReleasePD(
    float bodyUpX, float bodyUpZ, float angularVelocityX, float angularVelocityZ);
void DecayReleasedPlayerDifferential(
    std::array<float, 4>& playerThrust, float dt, float releaseBlend);
