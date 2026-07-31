// Preprocessor.hpp
#pragma once
#include <opencv2/core.hpp>

struct Letterboxed { cv::Mat img; float scale; int pad_left, pad_top; };
struct Preprocessed { cv::Mat blob; float scale; int pad_left, pad_top; };

Letterboxed  letterbox(const cv::Mat& src, int W = 640, int H = 640);
Preprocessed preprocess(const cv::Mat& src, int W = 640, int H = 640);
