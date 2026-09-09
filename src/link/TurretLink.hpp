#pragma once
#include <cstdint>
#include <string>

class TurretLink {
public:
    explicit TurretLink(std::string const& dev);
    ~TurretLink();

    bool ok() const { return fd_ >= 0; }

    // Absolute target in degrees. Returns false if nothing was sent
    // (link down, inside deadzone, or rate limited).
    bool aim(float panDeg, float tiltDeg);

    void home();

    float lastPan()  const { return lastPan_; }
    float lastTilt() const { return lastTilt_; }

private:
    void send(uint8_t type, int16_t pan, int16_t tilt);

    int      fd_{-1};
    uint16_t seq_{0};
    float    lastPan_{0.0f};
    float    lastTilt_{0.0f};
    int      frameCount_{0};

    static constexpr float kDeadzoneDeg = 2.5f;
    static constexpr int   kSendEveryN  = 3;
};
