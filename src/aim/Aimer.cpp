#include "Aimer.hpp"
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace { constexpr float kRadToDeg = 57.29577951308232f; }

Aimer::Aimer(std::string const& calibPath) {
    cv::FileStorage fs(calibPath, cv::FileStorage::READ);
    if (!fs.isOpened())
        throw std::runtime_error("Aimer: cannot open " + calibPath);
    fs["K"] >> K_;          // <-- match the node names you wrote in calib.yml
    fs["dist"] >> dist_;
    if (K_.empty() || dist_.empty())
        throw std::runtime_error("Aimer: missing K or dist in " + calibPath);
}

cv::Point2f Aimer::toAngles(cv::Point2f const& pixel) const {
    std::vector<cv::Point2f> in{pixel}, out;
    cv::undistortPoints(in, out, K_, dist_);   // out = (X/Z, Y/Z), normalized
    return { std::atan(out[0].x) * kRadToDeg + panOffset_,
             std::atan(out[0].y) * kRadToDeg + tiltOffset_ };
}

void Aimer::setOffsets(float panDeg, float tiltDeg) {
    panOffset_  = panDeg;
    tiltOffset_ = tiltDeg;
}
