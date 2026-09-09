#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "detect/Detector.hpp"
#include "aim/Aimer.hpp"
#include "link/TurretLink.hpp"
#include "track/Tracker.hpp"
#include "profiler.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static prof::Profiler P(30, "Turret loop");

static void onMouse(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    auto const* aimer = static_cast<Aimer const*>(userdata);
    cv::Point2f a = aimer->toAngles(cv::Point2f(float(x), float(y)));
    std::printf("pixel (%d, %d) -> pan %.2f  tilt %.2f\n", x, y, a.x, a.y);
    std::fflush(stdout);
}

static std::string gstreamer_pipeline(int cw, int ch, int dw, int dh, int fps, int flip) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(cw) +
           ", height=(int)" + std::to_string(ch) + ", framerate=(fraction)" + std::to_string(fps) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip) +
           " ! video/x-raw, width=(int)" + std::to_string(dw) + ", height=(int)" + std::to_string(dh) +
           ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

// Fine-tuned balloon model: single class, id 0.
static bool isTarget(Detection const& d) {
    return d.class_id == 0;
}

int main() {
    P.install_signal_handler();

    // HEADLESS=1 skips every GUI call so this runs over SSH with no X.
    // SEND=1 starts in sending mode (no keyboard to press 't' when headless).
    bool const headless = std::getenv("HEADLESS") != nullptr;
    bool sending = std::getenv("SEND") != nullptr;

    std::string pipeline = gstreamer_pipeline(1280, 720, 1280, 720, 30, 0);
    std::cout << "Using pipeline:\n\t" << pipeline << "\n";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cout << "Failed to open camera.\n";
        return -1;
    }

    Detector det("/home/helios/Desktop/Turret/models/balloon.engine", 0.20f);
    Aimer aimer("/home/helios/Desktop/Turret/data/calib.yml");
    aimer.setOffsets(-5.0f, -10.0f);
    TurretLink link("/dev/ttyACM0");
    Tracker tracker(cv::Matx22f(0.0578f, 0.f, 0.f, 0.0358f), 100.f, 300.f);

    if (!headless) {
        cv::namedWindow("CSI Camera", cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback("CSI Camera", onMouse, &aimer);
    }

    std::ofstream log("/home/helios/Desktop/Turret/data/noise.csv");
    log << "frame,cx,cy\n";

    cv::Point2f prev(-1.f, -1.f);
    long frameNo = 0;
    float avgDt = 0.f;
    int gateMisses = 0;
    auto lastFrame = std::chrono::steady_clock::now();

    cv::Mat img;
    if (headless)
        std::cout << "Headless mode, sending=" << sending << ". Ctrl-C to stop.\n";
    else
        std::cout << "Hit ESC to exit\n";

    while (true) {
        PROF_FRAME(P);

        bool ok;
        { PROF_STAGE(P, "capture"); ok = cap.read(img); }
        if (!ok) {
            std::cout << "Capture read error\n";
            break;
        }

        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - lastFrame).count();
        lastFrame = now;
        avgDt = (avgDt == 0.f) ? dt : 0.95f * avgDt + 0.05f * dt;
        if (frameNo % 30 == 0)
            std::printf("dt=%.4f  fps=%.1f\n", avgDt, 1.f / avgDt);

        std::vector<Detection> dets;
        { PROF_STAGE(P, "detect"); dets = det.detect(img); }

        cv::Point2f c(-1.f, -1.f);
        int best = -1;
        {
            PROF_STAGE(P, "gate");

            // Target-class only, then nearest to the previous centroid.
            float bestD = 1e9f;
            for (size_t i = 0; i < dets.size(); ++i) {
                if (!isTarget(dets[i])) continue;
                if (prev.x < 0) { best = int(i); break; }   // first lock: highest conf
                cv::Point2f cc = dets[i].center();
                float d = std::hypot(cc.x - prev.x, cc.y - prev.y);
                if (d < bestD) { bestD = d; best = int(i); }
            }
            if (prev.x >= 0 && bestD > 40.f) {
                best = -1;
                if (++gateMisses > 10) {          // gave up on the old position
                    prev = cv::Point2f(-1.f, -1.f);
                    gateMisses = 0;
                }
            } else {
                gateMisses = 0;
            }

            if (best >= 0) {
                auto const& d = dets[best];
                c = d.center();
                log << frameNo << ',' << c.x << ',' << c.y << '\n';
                prev = c;
                if (!headless) {
                    cv::rectangle(img, cv::Point2f(d.x1, d.y1), cv::Point2f(d.x2, d.y2),
                                  cv::Scalar(0, 255, 0), 2);
                    cv::circle(img, c, 4, cv::Scalar(0, 0, 255), -1);
                }
            }
        }

        if (!tracker.ready()) {
            if (best >= 0) tracker.init(c);
        } else {
            { PROF_STAGE(P, "track");
              tracker.predict(dt);
              if (best >= 0) tracker.update(c);
            }

            if (tracker.stale()) {
                tracker.reset();
                prev = cv::Point2f(-1.f, -1.f);
            } else if (best >= 0) {
                cv::Point2f tp = tracker.position();

                cv::Point2f ang;
                { PROF_STAGE(P, "aim"); ang = aimer.toAngles(tp); }

                if (sending) {
                    PROF_STAGE(P, "link");
                    link.aim(ang.x, ang.y);
                }

                if (!headless) {
                    cv::circle(img, tp, 5, cv::Scalar(255, 0, 0), -1);
                    char buf[96];
                    std::snprintf(buf, sizeof(buf), "pan %+.1f  tilt %+.1f  %s",
                                  ang.x, ang.y, sending ? "SENDING" : "print-only");
                    cv::putText(img, buf, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                                0.7, sending ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 255), 2);
                }
            }
        }

        ++frameNo;

        if (!headless) {
            int key;
            {
                PROF_STAGE(P, "display");
                cv::imshow("CSI Camera", img);
                key = cv::waitKey(1) & 0xff;
            }
            if (key == 27) break;
            if (key == 't') { sending = !sending; std::printf("sending=%d\n", sending); }
            if (key == 'h') link.home();
        }
    }

    cap.release();
    if (!headless) cv::destroyAllWindows();

    P.report();
    P.to_json("/home/helios/Desktop/Turret/data/bench_baseline.json");
    return 0;
}
