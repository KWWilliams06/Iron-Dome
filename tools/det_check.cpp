#include <chrono>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include "detect/Detector.hpp"

int main(int argc, char** argv)
{
    std::string const enginePath = (argc > 1) ? argv[1] : "../models/yolo26n.engine";
    std::string const imagePath  = (argc > 2) ? argv[2] : "../data/test_0.jpg";

    Detector det(enginePath);
    cv::Mat src = cv::imread(imagePath);
    if (src.empty()) throw std::runtime_error("cannot read image");

    det.detect(src);   // warm-up: first call pays lazy CUDA init

    auto t0 = std::chrono::steady_clock::now();
    auto dets = det.detect(src);
    auto t1 = std::chrono::steady_clock::now();

    for (auto const& d : dets)
        std::cout << "conf=" << d.confidence << " cls=" << d.class_id
                  << " box=[" << d.x1 << "," << d.y1 << ","
                  << d.x2 << "," << d.y2 << "]\n";

    std::cout << "detect(): "
              << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << " ms\n";
    return 0;
}
