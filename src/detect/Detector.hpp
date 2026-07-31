#pragma once
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int   class_id;

    cv::Point2f center() const {
        return { (x1 + x2) * 0.5f, (y1 + y2) * 0.5f };
    }
};

class Detector {
public:
    explicit Detector(std::string const& enginePath, float confThreshold = 0.25f);
    ~Detector();

    Detector(Detector&&) noexcept;
    Detector& operator=(Detector&&) noexcept;
    Detector(Detector const&) = delete;
    Detector& operator=(Detector const&) = delete;

    std::vector<Detection> detect(cv::Mat const& frame);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
