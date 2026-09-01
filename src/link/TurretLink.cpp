#include "TurretLink.hpp"
#include "../../firmware/protocol.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

TurretLink::TurretLink(std::string const& dev) {
    fd_ = open(dev.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) { std::perror("TurretLink open"); return; }

    termios tty{};
    if (tcgetattr(fd_, &tty) != 0) { std::perror("tcgetattr"); close(fd_); fd_ = -1; return; }

    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);
    cfmakeraw(&tty);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) { std::perror("tcsetattr"); close(fd_); fd_ = -1; return; }

    // Opening toggles DTR, resetting the Uno into its bootloader.
    std::printf("TurretLink: waiting for Arduino reset...\n");
    sleep(2);
    tcflush(fd_, TCIOFLUSH);
    std::printf("TurretLink: ready on %s\n", dev.c_str());
}

TurretLink::~TurretLink() {
    if (fd_ >= 0) close(fd_);
}

void TurretLink::send(uint8_t type, int16_t pan, int16_t tilt) {
    CommandPacket c{};
    c.sync = TURRET_SYNC;
    c.type = type;
    c.pan  = pan;
    c.tilt = tilt;
    c.seq  = seq_++;
    c.reserved = 0;
    uint8_t* raw = reinterpret_cast<uint8_t*>(&c);
    c.checksum = turret_checksum(raw);
    if (write(fd_, raw, TURRET_FRAME_LEN) != TURRET_FRAME_LEN)
        std::perror("TurretLink write");
}

void TurretLink::home() {
    if (fd_ < 0) return;
    send(CMD_HOME, 0, 0);
    lastPan_ = lastTilt_ = 0.0f;
}

bool TurretLink::aim(float panDeg, float tiltDeg) {
    if (fd_ < 0) return false;

    if (++frameCount_ < kSendEveryN) return false;
    frameCount_ = 0;

    if (std::fabs(panDeg  - lastPan_)  < kDeadzoneDeg &&
        std::fabs(tiltDeg - lastTilt_) < kDeadzoneDeg)
        return false;

    send(CMD_AIM,
         static_cast<int16_t>(std::lround(panDeg  * 100.0f)),
         static_cast<int16_t>(std::lround(tiltDeg * 100.0f)));

    lastPan_  = panDeg;
    lastTilt_ = tiltDeg;
    return true;
}
