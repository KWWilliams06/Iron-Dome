#include "Preprocessor.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cmath>


// ---------------------------------------------------------------------------
// Aspect-ratio-preserving resize with centered constant padding.
//
// Mirrors Ultralytics' LetterBox transform. The rounding quirks below are not
// arbitrary style choices — they reproduce the reference implementation
// bit-for-bit, which is what makes a C++/Python agreement test possible.
// ---------------------------------------------------------------------------
Letterboxed letterbox(const cv::Mat& src, int W, int H) {
    // Guard the two assumptions the rest of the function silently relies on:
    // a non-empty 8-bit 3-channel image. A CV_8UC4 frame (some capture paths
    // hand back BGRA) would otherwise sail through and produce a 4-channel
    // blob that the network rejects with an unhelpful shape error.
    CV_Assert(!src.empty() && src.type() == CV_8UC3);

    // min() of the two candidate ratios => the image *fits inside* the box.
    // Using max() would fill the box but crop content off the edges.
    //
    // Note this permits r > 1.0: a source smaller than 640 gets upscaled.
    // Ultralytics' `scaleup=False` mode clamps with r = min(r, 1.0f) instead,
    // padding rather than enlarging. Irrelevant for a 1280x720 CSI feed
    // (always downscaling), but it matters if this is ever fed a small crop.
    float r = std::min(W / (float)src.cols, H / (float)src.rows);

    // Dimensions of the scaled content, before padding. lround, not a C-style
    // truncating cast: (int)(719 * 0.5f) == 359, round(719 * 0.5f) == 360.
    // Ultralytics rounds, so we round.
    int w1 = static_cast<int>(std::lround(src.cols * r));
    int h1 = static_cast<int>(std::lround(src.rows * r));

    // Total leftover space, halved. Kept as float — the fractional half is
    // exactly what the +/-0.1 trick below resolves.
    float dw = (W - w1) / 2.0f;
    float dh = (H - h1) / 2.0f;

    // The -0.1 / +0.1 nudge is Ultralytics' method for splitting an odd
    // leftover without losing a pixel. Worked through:
    //
    //   leftover 280 (even): dw = 140.0
    //       left  = lround(139.9) = 140
    //       right = lround(140.1) = 140      -> 280, symmetric
    //
    //   leftover 281 (odd):  dw = 140.5
    //       left  = lround(140.4) = 140
    //       right = lround(140.6) = 141      -> 281, extra pixel on the right
    //
    // Without the nudge, lround(140.5) rounds half-away-from-zero to 141 on
    // both sides, giving 282 total and a 641-pixel-wide image.
    //
    // The max(0, ...) clamps are defensive only — r = min(...) guarantees
    // w1 <= W and h1 <= H, so dw and dh are already non-negative.
    int left   = std::max(0, static_cast<int>(std::lround(dw - 0.1f)));
    int right  = std::max(0, static_cast<int>(std::lround(dw + 0.1f)));
    int top    = std::max(0, static_cast<int>(std::lround(dh - 0.1f)));
    int bottom = std::max(0, static_cast<int>(std::lround(dh + 0.1f)));

    cv::Mat resized, out;

    // INTER_LINEAR to match the reference. INTER_AREA is the textbook choice
    // for downscaling and genuinely looks better, but it produces different
    // pixels — and different pixels mean different logits mean a failed
    // agreement test. Match first, optimize later.
    cv::resize(src, resized, cv::Size(w1, h1), 0, 0, cv::INTER_LINEAR);

    // 114 grey on all channels. An arbitrary constant, but the model saw it
    // during training augmentation, so padding with black (0) is a domain
    // shift the network was never shown.
    //
    // copyMakeBorder allocates a fresh buffer rather than writing into a
    // pre-sized Mat — one allocation per frame. Fine for now; if this shows
    // up in profiling, hoist a reusable 640x640 destination and blit into an
    // ROI instead.
    cv::copyMakeBorder(resized, out, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // Only left/top are returned. Right/bottom padding sits past the content
    // and never shifts a coordinate, so it plays no part in the inverse.
    return { out, r, left, top };
}

// ---------------------------------------------------------------------------
// Letterbox, then pack into the NCHW float tensor cv::dnn::Net::setInput wants.
// ---------------------------------------------------------------------------
Preprocessed preprocess(const cv::Mat& src, int W, int H) {
    Letterboxed lb = letterbox(src, W, H);

    // blobFromImage does four things at once. Argument by argument:
    //
    //   scalefactor 1.0/255.0 -> 0..255 uint8 into 0..1 float, the range the
    //                            model was trained on.
    //   size cv::Size()       -> EMPTY MEANS "DO NOT RESIZE". This is load-
    //                            bearing. Passing Size(640,640) here would
    //                            work by accident (input is already 640x640)
    //                            but invites someone to later hand this an
    //                            unletterboxed frame, at which point it
    //                            silently applies the naive aspect-destroying
    //                            stretch this whole file exists to avoid.
    //   mean cv::Scalar()     -> no mean subtraction. YOLO normalizes by
    //                            scale alone; ImageNet-style mean/std belongs
    //                            to a different family of models.
    //   swapRB true           -> BGR (OpenCV's order) -> RGB (the model's).
    //                            Silent failure mode if wrong: detections
    //                            still appear, just noticeably worse.
    //   crop false            -> no center crop. Only meaningful when size is
    //                            non-empty; false is the correct no-op here.
    //   ddepth CV_32F         -> output element type.
    //
    // Also implicit: HWC interleaved -> CHW planar, and prepending the batch
    // dimension. Result is [1,3,640,640].
    //
    // blobFromImage deep-copies, so the blob does not alias lb.img and stays
    // valid after lb goes out of scope.
    cv::Mat blob = cv::dnn::blobFromImage(
        lb.img, 1.0/255.0, cv::Size(), cv::Scalar(),
        true, false, CV_32F);

    // Carry the transform scalars through untouched — postprocess needs them.
    return { blob, lb.scale, lb.pad_left, lb.pad_top };
}
