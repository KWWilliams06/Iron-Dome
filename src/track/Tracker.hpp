#pragma once
#include <opencv2/core.hpp>

class Tracker {
public:
    Tracker(cv::Matx22f R, float sigmaA, float sigmaV0);
    void init(cv::Point2f z);
    void predict(float dt);
    void update(cv::Point2f z);
    void reset();
    cv::Point2f position() const { return {x_(0), x_(1)}; }
    cv::Point2f velocity() const { return {x_(2), x_(3)}; }
    cv::Vec2f   innovation() const { return y_; }
    bool        ready() const { return initialized_; }
    bool stale() const { return missed_ > kMaxMissed; }
    int  missed() const { return missed_; }
private:
    static constexpr int kMaxMissed = 5;
    cv::Vec4f             x_;
    cv::Matx44f           P_;
    cv::Matx<float, 2, 4> H_;
    cv::Matx22f           R_;
    float                 sigmaA_;
    float                 sigmaV0_;
    cv::Vec2f             y_{0.f, 0.f};
    bool                  initialized_{false};
    int                   missed_{0};
};
