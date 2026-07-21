#include "drone.h"
#include "rlgl.h"
#include <cmath>

void Drone::Init(Vector3 spawnPos) {
    position     = spawnPos;
    velocity     = {0, 0, 0};
    orientation  = QuaternionIdentity();
    angularVel   = {0, 0, 0};
    alive        = true;
    distanceTraveled = 0;

    const Color colors[ROTOR_COUNT] = {RED, BLUE, GREEN, YELLOW};
    const Vector3 localPositions[ROTOR_COUNT] = {
        {-ARM_LENGTH, 0, -ARM_LENGTH},  // Q: front-left
        {+ARM_LENGTH, 0, -ARM_LENGTH},  // W: front-right
        {-ARM_LENGTH, 0, +ARM_LENGTH},  // A: rear-left
        {+ARM_LENGTH, 0, +ARM_LENGTH},  // S: rear-right
    };

    for (int i = 0; i < ROTOR_COUNT; i++) {
        playerThrust[i]     = 0.0f;
        appliedThrust[i]    = 0.0f;
        rotors[i].spinAngle = 0;
        rotors[i].localPos  = localPositions[i];
        rotors[i].color     = colors[i];
    }
}

void Drone::SetRotorInput(RotorID id, bool keyDown, float dt) {
    float& T = playerThrust[id];
    if (keyDown)
        T = fminf(T + THRUST_RAMP_UP * dt, MAX_THRUST);
    else
        T = fmaxf(T - THRUST_RAMP_DOWN * dt, 0.0f);
}

void Drone::UsePlayerThrust() {
    appliedThrust = playerThrust;
}

void Drone::Update(float dt) {
    // --- Body-frame torques ---
    // Each motor at r=(rx,0,rz) with force F=(0,T,0):
    //   τ = r × F = (-rz*T, 0, rx*T)
    float tauX = 0, tauZ = 0;
    float totalThrust = 0;

    for (int i = 0; i < ROTOR_COUNT; i++) {
        const float T  = appliedThrust[i];
        const float rx = rotors[i].localPos.x;
        const float rz = rotors[i].localPos.z;
        tauX += -rz * T;
        tauZ +=  rx * T;
        totalThrust += T;
    }

    // Reactive yaw: Q,S spin CW; W,A spin CCW (opposite diagonals)
    float tauY = K_YAW * (appliedThrust[ROTOR_FRONT_LEFT] + appliedThrust[ROTOR_REAR_RIGHT]
                         - appliedThrust[ROTOR_FRONT_RIGHT] - appliedThrust[ROTOR_REAR_LEFT]);

    // --- Angular acceleration and velocity (body frame) ---
    angularVel.x += (tauX / I_PITCH) * dt;
    angularVel.y += (tauY / I_YAW)   * dt;
    angularVel.z += (tauZ / I_ROLL)  * dt;

    float angDrag = 1.0f - ANG_DRAG * dt;
    angularVel.x *= angDrag;
    angularVel.y *= angDrag;
    angularVel.z *= angDrag;

    // --- Quaternion integration (body-frame ω, left-multiply) ---
    // dq/dt = 0.5 * q * Ω, where Ω = (ωx,ωy,ωz,0)
    Quaternion omegaQ = {angularVel.x, angularVel.y, angularVel.z, 0.0f};
    Quaternion dq = QuaternionMultiply(orientation, omegaQ);
    orientation.x += 0.5f * dq.x * dt;
    orientation.y += 0.5f * dq.y * dt;
    orientation.z += 0.5f * dq.z * dt;
    orientation.w += 0.5f * dq.w * dt;
    orientation = QuaternionNormalize(orientation);

    // --- Linear forces ---
    Vector3 bodyThrust  = {0, totalThrust, 0};
    Vector3 worldThrust = Vector3RotateByQuaternion(bodyThrust, orientation);

    Vector3 worldForce = {
        worldThrust.x,
        worldThrust.y - DRONE_MASS * GRAVITY,
        worldThrust.z
    };

    float invMass = 1.0f / DRONE_MASS;
    velocity.x += worldForce.x * invMass * dt;
    velocity.y += worldForce.y * invMass * dt;
    velocity.z += worldForce.z * invMass * dt;

    float linDrag = 1.0f - LIN_DRAG * dt;
    velocity.x *= linDrag;
    velocity.y *= linDrag;
    velocity.z *= linDrag;

    position.x += velocity.x * dt;
    position.y += velocity.y * dt;
    position.z += velocity.z * dt;

    // --- Spin animation & distance tracking ---
    for (int i = 0; i < ROTOR_COUNT; i++)
        rotors[i].spinAngle += (appliedThrust[i] / MAX_THRUST) * 30.0f * dt;

    float fwdDist = -position.z;
    if (fwdDist > distanceTraveled)
        distanceTraveled = fwdDist;
}

Vector3 Drone::GetRotorWorldPos(RotorID id) const {
    return Vector3Add(position, Vector3RotateByQuaternion(rotors[id].localPos, orientation));
}

Vector3 Drone::GetForwardDir() const {
    Vector3 bodyFwd  = {0, 0, -1};
    Vector3 worldFwd = Vector3RotateByQuaternion(bodyFwd, orientation);
    worldFwd.y = 0;  // project to XZ plane so camera never rolls with drone
    float len = sqrtf(worldFwd.x * worldFwd.x + worldFwd.z * worldFwd.z);
    if (len > 0.001f) { worldFwd.x /= len; worldFwd.z /= len; }
    else              { worldFwd = {0, 0, -1}; }
    return worldFwd;
}

float Drone::GetAltitude() const { return position.y; }

float Drone::GetTiltAngle() const {
    Vector3 worldUp = Vector3RotateByQuaternion({0, 1, 0}, orientation);
    float dot = fmaxf(-1.0f, fminf(1.0f, worldUp.y));
    return acosf(dot) * RAD2DEG;
}

// ---------------------------------------------------------------------------
//  Rendering
// ---------------------------------------------------------------------------

void Drone::Draw() const {
    Matrix rotMat = QuaternionToMatrix(orientation);

    rlPushMatrix();
    rlTranslatef(position.x, position.y, position.z);
    rlMultMatrixf(MatrixToFloat(rotMat));  // macro expands to (MatrixToFloatV(mat).v)

    // Central body
    DrawCube({0, 0, 0}, 0.15f, 0.06f, 0.15f, DARKGRAY);

    // Arms from center to each motor
    for (int i = 0; i < ROTOR_COUNT; i++)
        DrawCylinderEx({0, 0, 0}, rotors[i].localPos, 0.012f, 0.012f, 6, GRAY);

    // Motor hubs
    for (int i = 0; i < ROTOR_COUNT; i++)
        DrawSphere(rotors[i].localPos, 0.04f, rotors[i].color);

    // Rotor discs (semi-transparent, radius scales with thrust)
    BeginBlendMode(BLEND_ALPHA);
    for (int i = 0; i < ROTOR_COUNT; i++) {
        const Rotor& r   = rotors[i];
        float discR      = 0.10f + 0.10f * (appliedThrust[i] / MAX_THRUST);
        Vector3 discBot  = {r.localPos.x, r.localPos.y - 0.005f, r.localPos.z};
        Vector3 discTop  = {r.localPos.x, r.localPos.y + 0.005f, r.localPos.z};
        DrawCylinderEx(discBot, discTop, discR, discR, 16, Fade(r.color, 0.40f));
    }
    EndBlendMode();

    // Spinner lines (rotate in rotor plane around motor's local Y axis)
    for (int i = 0; i < ROTOR_COUNT; i++) {
        const Rotor& r = rotors[i];
        float discR    = 0.10f + 0.10f * (appliedThrust[i] / MAX_THRUST);
        float cx = r.localPos.x, cy = r.localPos.y, cz = r.localPos.z;
        float ca = cosf(r.spinAngle), sa = sinf(r.spinAngle);
        Vector3 p1 = {cx + discR * ca, cy, cz + discR * sa};
        Vector3 p2 = {cx - discR * ca, cy, cz - discR * sa};
        DrawLine3D(p1, p2, r.color);
    }

    rlPopMatrix();
}

void Drone::DrawHUDBars(int screenW, int screenH) const {
    const int barW    = 40;
    const int barH    = 100;
    const int padding = 12;
    const int totalW  = ROTOR_COUNT * barW + (ROTOR_COUNT - 1) * padding;
    int xStart = (screenW - totalW) / 2;
    int yBar   = screenH - barH - 50;

    static const char* labels[ROTOR_COUNT] = {"Q", "W", "A", "S"};

    for (int i = 0; i < ROTOR_COUNT; i++) {
        int x = xStart + i * (barW + padding);

        DrawRectangle(x, yBar, barW, barH, {30, 30, 30, 200});
        DrawRectangleLines(x, yBar, barW, barH, DARKGRAY);

        int fillH = (int)(barH * appliedThrust[i] / MAX_THRUST);
        if (fillH > 0)
            DrawRectangle(x, yBar + barH - fillH, barW, fillH, rotors[i].color);

        // Key label centered below bar
        int labelW = MeasureText(labels[i], 20);
        DrawText(labels[i], x + (barW - labelW) / 2, yBar + barH + 6, 20, rotors[i].color);
    }
}
