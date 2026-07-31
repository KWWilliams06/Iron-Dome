#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

int main()
{
    int const BOARD_COLS = 9;        // inner corners, not squares
    int const BOARD_ROWS = 6;
    float const SQUARE_SIZE = 22.225f;   // mm

    // The 3D grid — identical for every image, since the board is planar
    // and we define its own coordinate system with z = 0.
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < BOARD_ROWS; ++i)
        for (int j = 0; j < BOARD_COLS; ++j)
            objp.emplace_back(j * SQUARE_SIZE, i * SQUARE_SIZE, 0.0f);

    std::vector<std::string> files;
    cv::glob("/home/helios/Desktop/Turret/data/calib/*.jpg", files); 
    if (files.empty()) { std::cerr << "no images found\n"; return 1; }

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    cv::Size imageSize;

    for (auto const& f : files) {
        cv::Mat img = cv::imread(f);
        if (img.empty()) continue;
        imageSize = img.size();

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(
            gray, cv::Size(BOARD_COLS, BOARD_ROWS), corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        std::cout << f << (found ? "  OK\n" : "  no board found\n");
        if (!found) continue;

        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                             30, 0.001));

        imagePoints.push_back(corners);
        objectPoints.push_back(objp);
    }

    std::cout << "usable images: " << imagePoints.size() << " / "
              << files.size() << "\n";
    if (imagePoints.size() < 10) {
        std::cerr << "too few usable images — recapture\n";
        return 1;
    }

    cv::Mat K, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize,
                                     K, distCoeffs, rvecs, tvecs);

    std::cout << "RMS reprojection error: " << rms << " px\n";
    std::cout << "fx=" << K.at<double>(0,0) << "  fy=" << K.at<double>(1,1)
              << "\ncx=" << K.at<double>(0,2) << "  cy=" << K.at<double>(1,2)
              << "\ndist=" << distCoeffs.t() << "\n";

     cv::FileStorage fs("/home/helios/Desktop/Turret/data/calib.yml", cv::FileStorage::WRITE);

     if (!fs.isOpened()) { std::cerr << "cannot write calib.yml\n"; return 1; }

     fs << "image_width"  << imageSize.width
       << "image_height" << imageSize.height
       << "K" << K
       << "dist" << distCoeffs
       << "rms" << rms;
    fs.release();
    std::cout << "wrote data/calib.yml\n";
    return 0;
}
