// Keyboard jog tool for the turret motor link.
//   a / d : pan  -/+
//   w / s : tilt +/-
//   h     : home (redefine current pan position as 0)
//   q     : quit
//
// Usage: ./build/serial_check /dev/ttyACM0

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include "../firmware/protocol.h"

static int openPort(const char *dev) {
    int fd = open(dev, O_RDWR | O_NOCTTY);
    if (fd < 0) { perror("open"); return -1; }

    termios tty{};
    if (tcgetattr(fd, &tty) != 0) { perror("tcgetattr"); close(fd); return -1; }

    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);
    cfmakeraw(&tty);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 5;   // 0.5s read timeout

    if (tcsetattr(fd, TCSANOW, &tty) != 0) { perror("tcsetattr"); close(fd); return -1; }

    printf("waiting for Arduino reset...\n");
    sleep(2);
    tcflush(fd, TCIOFLUSH);
    return fd;
}

static void sendCmd(int fd, uint8_t type, int16_t pan, int16_t tilt, uint16_t seq) {
    CommandPacket c{};
    c.sync = TURRET_SYNC;
    c.type = type;
    c.pan  = pan;
    c.tilt = tilt;
    c.seq  = seq;
    c.reserved = 0;
    uint8_t *raw = (uint8_t *)&c;
    c.checksum = turret_checksum(raw);
    if (write(fd, raw, TURRET_FRAME_LEN) != TURRET_FRAME_LEN)
        perror("write");
}

static void readTelemetry(int fd, int16_t *pan, int16_t *tilt) {
    uint8_t buf[TURRET_FRAME_LEN];
    ssize_t got = 0;
    while (got < TURRET_FRAME_LEN) {
        ssize_t n = read(fd, buf + got, TURRET_FRAME_LEN - got);
        if (n <= 0) { printf("  (no telemetry)\n"); return; }
        got += n;
    }
    TelemetryPacket *t = (TelemetryPacket *)buf;
    if (t->checksum != turret_checksum(buf)) { printf("  (bad checksum)\n"); return; }
    printf("  type=0x%02X pan=%.2f tilt=%.2f seq=%u flags=0x%02X\n",
           t->type, t->pan / 100.0, t->tilt / 100.0, t->seq, t->flags);

    if (t->type == TLM_ERROR && (t->flags & ERR_LIMIT_CLAMP)) {
        *pan  = t->pan;
        *tilt = t->tilt;
        printf("  (clamped, synced to %.2f / %.2f)\n", *pan / 100.0, *tilt / 100.0);
    }
}

int main(int argc, char **argv) {
    const char *dev = (argc > 1) ? argv[1] : "/dev/ttyACM0";
    int fd = openPort(dev);
    if (fd < 0) return 1;

    termios oldt{}, newt{};
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    int16_t pan = 0, tilt = 0;
    uint16_t seq = 0;
    const int16_t STEP = 500;   // 5.00 degrees per keypress

    printf("a/d = pan, w/s = tilt, h = home, q = quit\n");
    sendCmd(fd, CMD_PING, 0, 0, seq++);
    readTelemetry(fd, &pan, &tilt);

    bool running = true;
    while (running) {
        char k;
        if (read(STDIN_FILENO, &k, 1) != 1) continue;
        switch (k) {
            case 'a': pan  -= STEP; break;
            case 'd': pan  += STEP; break;
            case 'w': tilt += STEP; break;
            case 's': tilt -= STEP; break;
            case 'h': pan = 0; tilt = 0;
                      sendCmd(fd, CMD_HOME, 0, 0, seq++);
                      readTelemetry(fd, &pan, &tilt);
                      continue;
            case 'q': running = false; continue;
            default: continue;
        }
        printf("cmd pan=%.2f tilt=%.2f\n", pan / 100.0, tilt / 100.0);
        sendCmd(fd, CMD_AIM, pan, tilt, seq++);
        readTelemetry(fd, &pan, &tilt);
    }

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    close(fd);
    return 0;
}
