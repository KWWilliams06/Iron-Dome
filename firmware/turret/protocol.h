#ifndef TURRET_PROTOCOL_H
#define TURRET_PROTOCOL_H

#include <stdint.h>

// Wire format between Jetson (host) and Arduino (motor controller).
// Raw serial, 115200 baud, fixed 10-byte frames in both directions.
// Angles are int16 centidegrees: 4500 == 45.00 degrees. Range +/-327.67 deg.
// Compiles under both avr-gcc and aarch64 gcc.

#define TURRET_SYNC       0xA5
#define TURRET_FRAME_LEN  10

// Command types (host -> Arduino)
#define CMD_AIM           0x01  // absolute pan/tilt target
#define CMD_HOME          0x02  // reset pan position variable to 0
#define CMD_DISABLE       0x03  // de-energize stepper, detach servo
#define CMD_ENABLE        0x04  // re-energize
#define CMD_PING          0x05  // request telemetry

// Telemetry types (Arduino -> host)
#define TLM_STATE         0x81  // current pan/tilt and status
#define TLM_ERROR         0x82

// Status bit flags
#define ST_ENABLED        0x01
#define ST_PAN_MOVING     0x02
#define ST_TILT_MOVING    0x04
#define ST_HOMED          0x08

// Error bit flags
#define ERR_BAD_CHECKSUM  0x01
#define ERR_BAD_SYNC      0x02
#define ERR_LIMIT_CLAMP   0x04
#define ERR_UNKNOWN_CMD   0x08

// Mechanical limits, centidegrees
#define PAN_MIN_CDEG    (-9000)
#define PAN_MAX_CDEG     (9000)
#define TILT_MIN_CDEG   (-3000)
#define TILT_MAX_CDEG    (6000)

#pragma pack(push, 1)

typedef struct {
    uint8_t sync;      // always TURRET_SYNC
    uint8_t type;      // CMD_*
    int16_t pan;       // centidegrees, absolute
    int16_t tilt;      // centidegrees, absolute
    uint16_t seq;      // incrementing, for drop detection
    uint8_t reserved;
    uint8_t checksum;  // XOR of bytes 0..8
} CommandPacket;

typedef struct {
    uint8_t sync;
    uint8_t type;      // TLM_*
    int16_t pan;       // actual position, centidegrees
    int16_t tilt;
    uint16_t seq;      // echoes last command seq
    uint8_t flags;     // ST_* or ERR_* depending on type
    uint8_t checksum;
} TelemetryPacket;

#pragma pack(pop)

static inline uint8_t turret_checksum(const uint8_t *buf) {
    uint8_t c = 0;
    for (uint8_t i = 0; i < TURRET_FRAME_LEN - 1; i++) c ^= buf[i];
    return c;
}

#endif
