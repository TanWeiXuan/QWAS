#include "stability_assist.h"

#include "generated/stability_assist_weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace {
constexpr unsigned char kMagic[8] = {'Q', 'W', 'A', 'S', 'M', 'L', 'P', '\0'};
constexpr std::uint32_t kFormatVersion = 1;
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
    if (!std::equal(std::begin(kMagic), std::end(kMagic), bytes.begin())) {
        if (error) *error = "invalid magic bytes";
        return false;
    }
    const std::uint32_t expectedDimensions[5] = {
        kFormatVersion, STABILITY_ASSIST_OBSERVATION_SIZE, STABILITY_ASSIST_HIDDEN_SIZE,
        STABILITY_ASSIST_HIDDEN_SIZE, STABILITY_ASSIST_ACTION_SIZE};
    for (int i = 0; i < 5; ++i) {
        if (ReadU32(bytes.data() + 8 + i * sizeof(std::uint32_t)) != expectedDimensions[i]) {
            if (error) *error = i == 0 ? "unsupported format version" : "unexpected model dimensions";
            return false;
        }
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

float ComputeStabilityInterventionGate(float tiltRadians, float pitchRollRate) {
    constexpr float kPi = 3.14159265358979323846f;
    float tiltGate = SmoothStep(20.0f * kPi / 180.0f, 60.0f * kPi / 180.0f, tiltRadians);
    float rateGate = SmoothStep(1.5f, 5.0f, pitchRollRate);
    return std::max(tiltGate, rateGate);
}
