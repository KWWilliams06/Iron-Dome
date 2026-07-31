#include "aim/Aimer.hpp"
#include <cstdio>

int main() {
    Aimer aimer("/home/helios/Desktop/Turret/data/calib.yml");
    cv::Point2f pts[] = {{602.28f, 358.83f},   // optical center -> 0, 0
                         {640.0f,  360.0f},    // geometric center
                         {1279.0f, 360.0f},    // right edge, mid-height
                         {0.0f,    360.0f}};   // left edge
    for (auto const& p : pts) {
        cv::Point2f a = aimer.toAngles(p);
        std::printf("(%7.2f, %7.2f) -> pan %7.2f  tilt %7.2f\n", p.x, p.y, a.x, a.y);
    }
}
