#include "preprocess/Preprocessor.hpp"

#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: preprocess_check <in.jpg> <out_prefix>\n";
        return 1;
    }

    // BGR uint8 — same format cap.read() yields, so preprocess() is unchanged.
    cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "read failed: " << argv[1] << "\n";
        return 1;
    }

    const std::string prefix = argv[2];

    // Stage 1 — letterboxed uint8. PNG because it's lossless; JPEG here would
    // destroy the byte-exactness the whole test is checking for.
    Letterboxed lb = letterbox(src);
    cv::imwrite(prefix + "_lb.png", lb.img);

    // Stage 2 — the float32 NCHW blob, raw. Python reads it with
    // np.fromfile(path, dtype=np.float32).reshape(1, 3, 640, 640)
    Preprocessed pp = preprocess(src);
    std::ofstream f(prefix + "_blob.bin", std::ios::binary);
    if (!f) {
        std::cerr << "cannot open output: " << prefix << "_blob.bin\n";
        return 1;
    }
    f.write(reinterpret_cast<const char*>(pp.blob.ptr<float>()),
            pp.blob.total() * sizeof(float));

    // Inverse-transform scalars, for the Python side to check too.
    std::cout << src.cols << " " << src.rows << " "
              << pp.scale << " " << pp.pad_left << " " << pp.pad_top << "\n";
    return 0;
}
