#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <string>

#ifndef REPO_ROOT
#define REPO_ROOT "/home/helios/Desktop/Turret"
#endif

static termios g_old;

static void raw_mode_on() {
    tcgetattr(STDIN_FILENO, &g_old);
    termios raw = g_old;
    raw.c_lflag &= ~(ICANON | ECHO);      // no line buffering, no echo
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    fcntl(STDIN_FILENO, F_SETFL, fcntl(STDIN_FILENO, F_GETFL) | O_NONBLOCK);
}

static void raw_mode_off() {
    tcsetattr(STDIN_FILENO, TCSANOW, &g_old);
}

static std::string gstreamer_pipeline(int cw, int ch, int dw, int dh, int fps, int flip) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(cw) +
           ", height=(int)" + std::to_string(ch) + ", framerate=(fraction)" + std::to_string(fps) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip) +
           " ! video/x-raw, width=(int)" + std::to_string(dw) + ", height=(int)" + std::to_string(dh) +
           ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: capture_dataset <prefix> [start_index]\n");
        return 1;
    }
    std::string prefix = argv[1];
    int saved = (argc > 2) ? std::atoi(argv[2]) : 0;

    std::string dir = std::string(REPO_ROOT) + "/data/balloon/raw";
    std::string mk  = "mkdir -p " + dir;
    if (std::system(mk.c_str()) != 0) {
        std::printf("could not create %s\n", dir.c_str());
        return 1;
    }

    cv::VideoCapture cap(gstreamer_pipeline(1280, 720, 1280, 720, 30, 0), cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::printf("camera open failed\n");
        return 1;
    }

    cv::Mat img;
    for (int i = 0; i < 30; ++i) cap.read(img);   // let auto-exposure settle

    raw_mode_on();
    std::printf("s = save   q = quit\n");
    std::fflush(stdout);

    while (true) {
        if (!cap.read(img)) { std::printf("read error\r\n"); break; }

        char ch;
        if (::read(STDIN_FILENO, &ch, 1) != 1) continue;   // nothing pressed
        if (ch == 'q' || ch == 27) break;
        if (ch != 's') continue;

        cv::Scalar m = cv::mean(img);
        double bright = (m[0] + m[1] + m[2]) / 3.0;

        std::string path = dir + "/" + prefix + "_" + std::to_string(saved) + ".jpg";
        if (!cv::imwrite(path, img)) { std::printf("write failed: %s\r\n", path.c_str()); break; }

        std::printf("[%3d] %s_%d.jpg  mean=%.1f\r\n", saved + 1, prefix.c_str(), saved, bright);
        std::fflush(stdout);
        ++saved;
    }

    raw_mode_off();
    cap.release();
    std::printf("done: %d frames in %s\n", saved, dir.c_str());
    return 0;
}
