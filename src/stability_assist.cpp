#include "stability_assist.h"

#include "drone.h"
#include "generated/stability_assist_weights.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace {
constexpr unsigned char kMagic[8] = {'Q', 'W', 'A', 'S', 'M', 'L', 'P', '\0'};
constexpr std::uint32_t kFormatVersion = 2;
constexpr std::size_t kHeaderBytes = 8 + 5 * sizeof(std::uint32_t);
constexpr std::size_t kFloatCount =
    STABILITY_ASSIST_OBSERVATION_SIZE +
    STABILITY_ASSIST_HIDDEN_SIZE * STABILITY_ASSIST_OBSERVATION_SIZE +
    STABILITY_ASSIST_HIDDEN_SIZE +
    STABILITY_ASSIST_HIDDEN_SIZE * STABILITY_ASSIST_HIDDEN_SIZE +
    STABILITY_ASSIST_HIDDEN_SIZE +
    STABILITY_ASSIST_ACTION_SIZE * STABILITY_ASSIST_HIDDEN_SIZE +
    STABILITY_ASSIST_ACTION_SIZE;
constexpr std::size_t kFileBytes = kHeaderBytes + kFloatCount * sizeof(float);

float SmoothStep(float edge0, float edge1, float x) {
    float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

std::uint32_t ReadU32(const unsigned char* bytes) {
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8u) |
           (static_cast<std::uint32_t>(bytes[2]) << 16u) |
           (static_cast<std::uint32_t>(bytes[3]) << 24u);
}

float ReadF32(const unsigned char* bytes) {
    std::uint32_t bits = ReadU32(bytes);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

template <std::size_t N>
bool ReadFloatArray(const std::vector<unsigned char>& bytes, std::size_t& offset,
                    std::array<float, N>& destination, bool requirePositive,
                    std::string* error) {
    for (float& value : destination) {
        value = ReadF32(bytes.data() + offset);
        offset += sizeof(float);
        if (!std::isfinite(value) || (requirePositive && value <= 0.0f)) {
            if (error) *error = requirePositive ? "invalid observation scale" : "non-finite model value";
            return false;
        }
    }
    return true;
}
}  // namespace

StabilityAssistMLP::StabilityAssistMLP() {
    UseBakedWeights();
}

void StabilityAssistMLP::UseBakedWeights() {
    static_assert(qwas_baked_stability_assist::kFormatVersion == kFormatVersion);
    static_assert(qwas_baked_stability_assist::kInputSize == STABILITY_ASSIST_OBSERVATION_SIZE);
    static_assert(qwas_baked_stability_assist::kHidden1Size == STABILITY_ASSIST_HIDDEN_SIZE);
    static_assert(qwas_baked_stability_assist::kHidden2Size == STABILITY_ASSIST_HIDDEN_SIZE);
    static_assert(qwas_baked_stability_assist::kOutputSize == STABILITY_ASSIST_ACTION_SIZE);
    observationScale_ = qwas_baked_stability_assist::kObservationScale;
    weights1_ = qwas_baked_stability_assist::kWeights1;
    bias1_ = qwas_baked_stability_assist::kBias1;
    weights2_ = qwas_baked_stability_assist::kWeights2;
    bias2_ = qwas_baked_stability_assist::kBias2;
    weights3_ = qwas_baked_stability_assist::kWeights3;
    bias3_ = qwas_baked_stability_assist::kBias3;
}

StabilityAssistMLP::Action StabilityAssistMLP::Forward(const Observation& rawObservation) const {
    std::array<float, STABILITY_ASSIST_OBSERVATION_SIZE> observation{};
    std::array<float, STABILITY_ASSIST_HIDDEN_SIZE> hidden1{};
    std::array<float, STABILITY_ASSIST_HIDDEN_SIZE> hidden2{};
    Action action{};

    for (int i = 0; i < STABILITY_ASSIST_OBSERVATION_SIZE; ++i) {
        if (!std::isfinite(rawObservation[i]) || !std::isfinite(observationScale_[i]) || observationScale_[i] <= 0.0f)
            return {};
        observation[i] = std::clamp(rawObservation[i] / observationScale_[i], -4.0f, 4.0f);
    }

    for (int output = 0; output < STABILITY_ASSIST_HIDDEN_SIZE; ++output) {
        float sum = bias1_[output];
        for (int input = 0; input < STABILITY_ASSIST_OBSERVATION_SIZE; ++input)
            sum += weights1_[output * STABILITY_ASSIST_OBSERVATION_SIZE + input] * observation[input];
        hidden1[output] = std::tanh(sum);
    }
    for (int output = 0; output < STABILITY_ASSIST_HIDDEN_SIZE; ++output) {
        float sum = bias2_[output];
        for (int input = 0; input < STABILITY_ASSIST_HIDDEN_SIZE; ++input)
            sum += weights2_[output * STABILITY_ASSIST_HIDDEN_SIZE + input] * hidden1[input];
        hidden2[output] = std::tanh(sum);
    }
    for (int output = 0; output < STABILITY_ASSIST_ACTION_SIZE; ++output) {
        float sum = bias3_[output];
        for (int input = 0; input < STABILITY_ASSIST_HIDDEN_SIZE; ++input)
            sum += weights3_[output * STABILITY_ASSIST_HIDDEN_SIZE + input] * hidden2[input];
        action[output] = std::tanh(sum);
        if (!std::isfinite(action[output]))
            return {};
    }
    return action;
}

bool StabilityAssistMLP::LoadFromFile(const std::string& path, std::string* error) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        if (error) *error = "could not open file";
        return false;
    }
    const std::streamsize length = stream.tellg();
    if (length < static_cast<std::streamsize>(kHeaderBytes)) {
        if (error) *error = "unexpected file length";
        return false;
    }
    stream.seekg(0);
    std::array<unsigned char, kHeaderBytes> header{};
    if (!stream.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()))) {
        if (error) *error = "could not read file";
        return false;
    }
    if (!std::equal(std::begin(kMagic), std::end(kMagic), header.begin())) {
        if (error) *error = "invalid magic bytes";
        return false;
    }
    const std::uint32_t expectedDimensions[5] = {
        kFormatVersion, STABILITY_ASSIST_OBSERVATION_SIZE, STABILITY_ASSIST_HIDDEN_SIZE,
        STABILITY_ASSIST_HIDDEN_SIZE, STABILITY_ASSIST_ACTION_SIZE};
    for (int i = 0; i < 5; ++i) {
        if (ReadU32(header.data() + 8 + i * sizeof(std::uint32_t)) != expectedDimensions[i]) {
            if (error) *error = i == 0 ? "incompatible model format version (expected 2)" : "unexpected model dimensions";
            return false;
        }
    }
    if (length != static_cast<std::streamsize>(kFileBytes)) {
        if (error) *error = "unexpected file length";
        return false;
    }
    stream.seekg(0);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(length));
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), length)) {
        if (error) *error = "could not read file";
        return false;
    }

    decltype(observationScale_) observationScale;
    decltype(weights1_) weights1;
    decltype(bias1_) bias1;
    decltype(weights2_) weights2;
    decltype(bias2_) bias2;
    decltype(weights3_) weights3;
    decltype(bias3_) bias3;
    std::size_t offset = kHeaderBytes;
    if (!ReadFloatArray(bytes, offset, observationScale, true, error) ||
        !ReadFloatArray(bytes, offset, weights1, false, error) ||
        !ReadFloatArray(bytes, offset, bias1, false, error) ||
        !ReadFloatArray(bytes, offset, weights2, false, error) ||
        !ReadFloatArray(bytes, offset, bias2, false, error) ||
        !ReadFloatArray(bytes, offset, weights3, false, error) ||
        !ReadFloatArray(bytes, offset, bias3, false, error))
        return false;

    observationScale_ = observationScale;
    weights1_ = weights1;
    bias1_ = bias1;
    weights2_ = weights2;
    bias2_ = bias2;
    weights3_ = weights3;
    bias3_ = bias3;
    return true;
}

float ComputeStabilityDangerGate(float tiltRadians, float pitchRollRate) {
    constexpr float kPi = 3.14159265358979323846f;
    float tiltGate = SmoothStep(20.0f * kPi / 180.0f, 60.0f * kPi / 180.0f, tiltRadians);
    float rateGate = SmoothStep(1.5f, 5.0f, pitchRollRate);
    return std::max(tiltGate, rateGate);
}

float ComputeStabilityReleaseRecoveryGate(float tiltRadians, float pitchRollRate) {
    constexpr float kPi = 3.14159265358979323846f;
    float tiltGate = SmoothStep(1.5f * kPi / 180.0f, 20.0f * kPi / 180.0f, tiltRadians);
    float rateGate = SmoothStep(0.12f, 2.0f, pitchRollRate);
    return std::max(tiltGate, rateGate);
}

float ComputeStabilityReleaseBlend(float timeSincePlayerInput) {
    return SmoothStep(EASY_MODE_RELEASE_BLEND_START, EASY_MODE_RELEASE_BLEND_FULL, timeSincePlayerInput);
}

std::array<float, 2> ComputeStabilityReleasePD(
    float bodyUpX, float bodyUpZ, float angularVelocityX, float angularVelocityZ) {
    return {
        -EASY_MODE_RELEASE_PITCH_KP * bodyUpZ - EASY_MODE_RELEASE_PITCH_KD * angularVelocityX,
        +EASY_MODE_RELEASE_ROLL_KP * bodyUpX - EASY_MODE_RELEASE_ROLL_KD * angularVelocityZ,
    };
}

void DecayReleasedPlayerDifferential(
    std::array<float, 4>& playerThrust, float dt, float releaseBlend) {
    float mean = 0.25f * (playerThrust[0] + playerThrust[1] + playerThrust[2] + playerThrust[3]);
    float exponentialDecay = std::exp(-dt / EASY_MODE_RELEASE_DIFFERENTIAL_TIME_CONSTANT);
    float blendedDecay = 1.0f - std::clamp(releaseBlend, 0.0f, 1.0f) * (1.0f - exponentialDecay);
    for (float& thrust : playerThrust)
        thrust = std::max(0.0f, mean + (thrust - mean) * blendedDecay);
}

void StabilityAssistController::Reset() {
    timeSincePlayerInput_ = 0.0f;
    releaseBlend_ = 0.0f;
    previousPitchResidual_ = 0.0f;
    previousRollResidual_ = 0.0f;
}

StabilityAssistTelemetry StabilityAssistController::Apply(
    Drone& drone,
    const std::array<bool, 4>& playerButtons,
    float dt,
    const StabilityAssistMLP& model) {
    const bool anyButtonHeld = playerButtons[0] || playerButtons[1] || playerButtons[2] || playerButtons[3];
    if (anyButtonHeld)
        timeSincePlayerInput_ = 0.0f;
    else
        timeSincePlayerInput_ += dt;
    releaseBlend_ = ComputeStabilityReleaseBlend(timeSincePlayerInput_);

    if (!anyButtonHeld)
        DecayReleasedPlayerDifferential(drone.playerThrust, dt, releaseBlend_);

    float playerTotal = 0.0f;
    for (float thrust : drone.playerThrust)
        playerTotal += thrust;
    float sinkTarget = EASY_MODE_SINK_THRUST_RATIO * DRONE_MASS * GRAVITY;
    float collectiveTopUp = std::max(0.0f, sinkTarget - playerTotal) / ROTOR_COUNT;
    for (float thrust : drone.playerThrust)
        collectiveTopUp = std::min(collectiveTopUp, std::max(0.0f, MAX_THRUST - thrust));

    Vector3 bodyUp = Vector3RotateByQuaternion({0, 1, 0}, drone.orientation);
    float tiltRadians = std::acos(std::clamp(bodyUp.y, -1.0f, 1.0f));
    float pitchRollRate = std::sqrt(drone.angularVel.x * drone.angularVel.x +
                                    drone.angularVel.z * drone.angularVel.z);
    float dangerGate = ComputeStabilityDangerGate(tiltRadians, pitchRollRate);
    float releaseRecoveryGate = ComputeStabilityReleaseRecoveryGate(tiltRadians, pitchRollRate);
    float effectiveGate = std::max(dangerGate, releaseBlend_ * releaseRecoveryGate);

    StabilityAssistMLP::Observation observation{};
    observation[ASSIST_BODY_UP_X] = bodyUp.x;
    observation[ASSIST_BODY_UP_Y] = bodyUp.y;
    observation[ASSIST_BODY_UP_Z] = bodyUp.z;
    observation[ASSIST_ANGULAR_VELOCITY_X] = drone.angularVel.x;
    observation[ASSIST_ANGULAR_VELOCITY_Y] = drone.angularVel.y;
    observation[ASSIST_ANGULAR_VELOCITY_Z] = drone.angularVel.z;
    for (int i = 0; i < ROTOR_COUNT; ++i) {
        observation[ASSIST_PLAYER_THRUST_FRONT_LEFT + i] =
            MAX_THRUST > 0.0f ? drone.playerThrust[i] / MAX_THRUST : 0.0f;
        observation[ASSIST_BUTTON_FRONT_LEFT + i] = playerButtons[i] ? 1.0f : 0.0f;
    }
    observation[ASSIST_FRAME_DT] = dt;
    observation[ASSIST_INTERVENTION_GATE] = effectiveGate;
    const float physics[] = {DRONE_MASS, GRAVITY, MAX_THRUST, THRUST_RAMP_UP, THRUST_RAMP_DOWN,
                             ARM_LENGTH, I_PITCH, I_YAW, I_ROLL, LIN_DRAG, ANG_DRAG, K_YAW};
    for (int i = 0; i < 12; ++i)
        observation[ASSIST_MASS + i] = physics[i];
    observation[ASSIST_RELEASE_BLEND] = releaseBlend_;
    observation[ASSIST_TIME_SINCE_PLAYER_INPUT] = timeSincePlayerInput_;
    observation[ASSIST_PREVIOUS_PITCH_RESIDUAL] = previousPitchResidual_;
    observation[ASSIST_PREVIOUS_ROLL_RESIDUAL] = previousRollResidual_;

    StabilityAssistMLP::Action residual = model.Forward(observation);
    auto pd = ComputeStabilityReleasePD(bodyUp.x, bodyUp.z, drone.angularVel.x, drone.angularVel.z);
    float activePitch = dangerGate * residual[0];
    float activeRoll = dangerGate * residual[1];
    float releasePitch = releaseRecoveryGate * (pd[0] + EASY_MODE_RELEASE_RESIDUAL_SCALE * residual[0]);
    float releaseRoll = releaseRecoveryGate * (pd[1] + EASY_MODE_RELEASE_RESIDUAL_SCALE * residual[1]);
    float pitchCommand = std::clamp((1.0f - releaseBlend_) * activePitch + releaseBlend_ * releasePitch, -1.0f, 1.0f);
    float rollCommand = std::clamp((1.0f - releaseBlend_) * activeRoll + releaseBlend_ * releaseRoll, -1.0f, 1.0f);
    float authorityRatio = (1.0f - releaseBlend_) * EASY_MODE_ACTIVE_DIFFERENTIAL_THRUST_RATIO +
                           releaseBlend_ * EASY_MODE_RELEASE_DIFFERENTIAL_THRUST_RATIO;

    constexpr float pitchMode[ROTOR_COUNT] = {1.0f, 1.0f, -1.0f, -1.0f};
    constexpr float rollMode[ROTOR_COUNT] = {-1.0f, 1.0f, -1.0f, 1.0f};
    std::array<float, ROTOR_COUNT> correction{};
    float feasibilityScale = 1.0f;
    for (int i = 0; i < ROTOR_COUNT; ++i) {
        drone.appliedThrust[i] = drone.playerThrust[i] + collectiveTopUp;
        correction[i] = (pitchCommand * pitchMode[i] + rollCommand * rollMode[i]) * authorityRatio * MAX_THRUST;
        if (correction[i] > 0.0f)
            feasibilityScale = std::min(feasibilityScale, (MAX_THRUST - drone.appliedThrust[i]) / correction[i]);
        else if (correction[i] < 0.0f)
            feasibilityScale = std::min(feasibilityScale, drone.appliedThrust[i] / -correction[i]);
    }
    feasibilityScale = std::clamp(feasibilityScale, 0.0f, 1.0f);
    float assistMagnitude = 0.0f;
    for (int i = 0; i < ROTOR_COUNT; ++i) {
        correction[i] *= feasibilityScale;
        assistMagnitude += std::fabs(correction[i]) / std::max(MAX_THRUST, 1.0e-6f);
        drone.appliedThrust[i] = std::clamp(drone.appliedThrust[i] + correction[i], 0.0f, MAX_THRUST);
    }

    previousPitchResidual_ = residual[0];
    previousRollResidual_ = residual[1];
    StabilityAssistTelemetry telemetry;
    telemetry.releaseBlend = releaseBlend_;
    telemetry.dangerGate = dangerGate;
    telemetry.releaseRecoveryGate = releaseRecoveryGate;
    telemetry.effectiveGate = effectiveGate;
    telemetry.pitchPD = pd[0];
    telemetry.rollPD = pd[1];
    telemetry.pitchResidual = residual[0];
    telemetry.rollResidual = residual[1];
    telemetry.pitchCommand = pitchCommand;
    telemetry.rollCommand = rollCommand;
    telemetry.assistMagnitude = assistMagnitude * 0.25f;
    telemetry.pdSaturated = std::fabs(pd[0]) > 1.0f || std::fabs(pd[1]) > 1.0f;
    return telemetry;
}
