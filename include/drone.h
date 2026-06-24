#pragma once
#include "raylib.h"
#include "raymath.h"

// Physics constants (SI units: meters, kg, seconds)
constexpr float DRONE_MASS       = 0.5f;
constexpr float GRAVITY          = 9.81f;
constexpr float MAX_THRUST       = 3.0f;   // N per motor; hover needs ~1.23N each
constexpr float THRUST_RAMP_UP   = 6.0f;   // N/s (0 → max in 0.5s)
constexpr float THRUST_RAMP_DOWN = 10.0f;  // N/s (max → 0 in 0.3s)
constexpr float ARM_LENGTH       = 0.25f;  // m (center to motor)
constexpr float I_PITCH          = 0.004f; // kg·m² (around body-X, right-wing axis)
constexpr float I_YAW            = 0.007f; // kg·m² (around body-Y, up axis)
constexpr float I_ROLL           = 0.004f; // kg·m² (around body-Z, tail axis)
constexpr float LIN_DRAG         = 0.4f;   // /s
constexpr float ANG_DRAG         = 2.0f;   // /s
constexpr float K_YAW            = 0.02f;  // reactive yaw coefficient

// Body frame convention:
//   body-X = right wing
//   body-Y = up (thrust direction)
//   body-Z = tail (backward; forward = -Z)
//
// Motor layout matching keyboard:
//   Q W    <- front (-Z side)
//   A S    <- rear (+Z side)

constexpr int ROTOR_COUNT = 4;
enum RotorID { ROTOR_FRONT_LEFT = 0, ROTOR_FRONT_RIGHT = 1, ROTOR_BACK_LEFT = 2, ROTOR_BACK_RIGHT = 3 };

struct Rotor {
    float   thrust;     // current thrust [0, MAX_THRUST] N
    float   spinAngle;  // accumulated radians (animation only)
    Vector3 localPos;   // offset from drone center in body frame
    Color   color;      // Q=RED, W=BLUE, A=GREEN, S=YELLOW
};

struct Drone {
    Vector3    position;
    Vector3    velocity;
    Quaternion orientation;  // body-from-world; init = QuaternionIdentity
    Vector3    angularVel;   // body frame rad/s
    Rotor      rotors[ROTOR_COUNT];
    bool       alive;
    float      distanceTraveled;  // max forward (-Z) distance reached

    void Init(Vector3 spawnPos);
    void SetRotorInput(RotorID id, bool keyDown, float dt);
    void Update(float dt);

    // 3D rendering — call inside BeginMode3D/EndMode3D
    void Draw() const;
    // 2D HUD thrust bars — call after EndMode3D
    void DrawHUDBars(int screenW, int screenH) const;

    Vector3 GetRotorWorldPos(RotorID id) const;
    Vector3 GetForwardDir() const;   // yaw-only XZ forward direction
    float   GetAltitude() const;
    float   GetTiltAngle() const;    // degrees from upright (0 = level)
};
