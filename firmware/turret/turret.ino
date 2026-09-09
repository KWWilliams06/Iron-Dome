// Turret motor controller -- Arduino Uno.
//
// Wiring:
//   Pan  : NEMA 17 via DRV8825.  D2 = DIR, D3 = STEP.  Motor rail 12V 3A.
//   Tilt : HD-2522MG servo.      D9 = signal.          Servo rail 6V 5A.
//   DRV8825 logic GND and servo supply GND both tie to Arduino GND.
//
// Host link: 115200 baud, 10-byte frames, see ../protocol.h

#include <Servo.h>
#include "protocol.h"

#define PIN_DIR    2
#define PIN_STEP   3
#define PIN_TILT   9

// --- Pan (stepper) ---------------------------------------------------
// 200 full steps/rev = 1.8 deg = 180 centidegrees per step.
// MODE pins are unwired, so the DRV8825 runs in full-step mode.
#define CDEG_PER_STEP     11
#define STEP_INTERVAL_US  300L   // max step rate; matches bench test
#define PAN_DIR_SIGN      (-1)       // flip to -1 if pan runs backwards

// --- Tilt (servo) ----------------------------------------------------
#define SERVO_CENTER_US   1500
#define US_PER_DEG        10.0f   // NEEDS CALIBRATION, see notes
#define TILT_SIGN         (-1)       // flip to -1 if tilt runs backwards

Servo tiltServo;

long panSteps      = 0;   // current pan position, in steps
long panTargetSteps = 0;
int  tiltCdeg      = 0;
uint16_t lastSeq   = 0;
uint8_t  status    = ST_HOMED;
unsigned long lastStepUs = 0;

uint8_t rxBuf[TURRET_FRAME_LEN];
uint8_t rxLen = 0;

static int16_t clampCdeg(int16_t v, int16_t lo, int16_t hi, uint8_t *err) {
  if (v < lo) { *err |= ERR_LIMIT_CLAMP; return lo; }
  if (v > hi) { *err |= ERR_LIMIT_CLAMP; return hi; }
  return v;
}

static void sendTelemetry(uint8_t type, uint8_t flags) {
  TelemetryPacket t;
  t.sync  = TURRET_SYNC;
  t.type  = type;
  t.pan   = (int16_t)((panSteps * 180) / 16);
  t.tilt  = tiltCdeg;
  t.seq   = lastSeq;
  t.flags = flags;
  uint8_t *raw = (uint8_t *)&t;
  t.checksum = turret_checksum(raw);
  Serial.write(raw, TURRET_FRAME_LEN);
}

static void applyTilt(int16_t cdeg) {
  tiltCdeg = cdeg;
  float deg = cdeg / 100.0f;
  int us = SERVO_CENTER_US + (int)(TILT_SIGN * deg * US_PER_DEG);
  tiltServo.writeMicroseconds(us);
}

static void handleFrame() {
  CommandPacket *c = (CommandPacket *)rxBuf;

  if (c->sync != TURRET_SYNC)                { sendTelemetry(TLM_ERROR, ERR_BAD_SYNC);     return; }
  if (c->checksum != turret_checksum(rxBuf)) { sendTelemetry(TLM_ERROR, ERR_BAD_CHECKSUM); return; }

  lastSeq = c->seq;
  uint8_t err = 0;

  switch (c->type) {
    case CMD_AIM: {
      int16_t p = clampCdeg(c->pan,  PAN_MIN_CDEG,  PAN_MAX_CDEG,  &err);
      int16_t t = clampCdeg(c->tilt, TILT_MIN_CDEG, TILT_MAX_CDEG, &err);
      panTargetSteps = ((long)p * 16) / 180;
      applyTilt(t);
      break;
    }
    case CMD_HOME:
      panSteps = 0;
      panTargetSteps = 0;
      status |= ST_HOMED;
      break;
    case CMD_DISABLE:
      tiltServo.detach();
      status &= ~ST_ENABLED;
      break;
    case CMD_ENABLE:
      tiltServo.attach(PIN_TILT);
      applyTilt(tiltCdeg);
      status |= ST_ENABLED;
      break;
    case CMD_PING:
      break;
    default:
      sendTelemetry(TLM_ERROR, ERR_UNKNOWN_CMD);
      return;
  }

  if (err) sendTelemetry(TLM_ERROR, err);
  else     sendTelemetry(TLM_STATE, status);
}

static void servicePan() {
  if (panSteps == panTargetSteps) {
    status &= ~ST_PAN_MOVING;
    return;
  }
  unsigned long now = micros();
  if (now - lastStepUs < STEP_INTERVAL_US) return;
  lastStepUs = now;

  status |= ST_PAN_MOVING;
  int dir = (panTargetSteps > panSteps) ? 1 : -1;
  digitalWrite(PIN_DIR, (dir * PAN_DIR_SIGN > 0) ? HIGH : LOW);
  digitalWrite(PIN_STEP, HIGH);
  delayMicroseconds(3);
  digitalWrite(PIN_STEP, LOW);
  panSteps += dir;
}

void setup() {
  pinMode(PIN_DIR, OUTPUT);
  pinMode(PIN_STEP, OUTPUT);
  digitalWrite(PIN_STEP, LOW);

  tiltServo.attach(PIN_TILT);
  applyTilt(0);
  status |= ST_ENABLED;

  Serial.begin(115200);
}

void loop() {
  while (Serial.available()) {
    uint8_t b = Serial.read();
    if (rxLen == 0 && b != TURRET_SYNC) continue;  // resync
    rxBuf[rxLen++] = b;
    if (rxLen == TURRET_FRAME_LEN) {
      handleFrame();
      rxLen = 0;
    }
  }
  servicePan();
}
