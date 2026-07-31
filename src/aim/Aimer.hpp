#pragma once
#include <string>
#include <opencv2/core.hpp>

class Aimer {
public:
    explicit Aimer(std::string const& calibPath);

    // Pixel coords -> (pan, tilt) in degrees relative to the optical axis.
    // +pan = target is right of center, +tilt = target is below center.
    cv::Point2f toAngles(cv::Point2f const& pixel) const;

    // Fixed camera-to-turret mounting misalignment, measured once the
    // hardware exists. Added to every toAngles() result.
    void setOffsets(float panDeg, float tiltDeg);

private:
    cv::Mat K_, dist_;
    float panOffset_{0.0f};
    float tiltOffset_{0.0f};
};

