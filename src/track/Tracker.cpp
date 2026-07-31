#include "Tracker.hpp"

Tracker::Tracker(cv::Matx22f R, float sigmaA, float sigmaV0)
    : x_(0.f, 0.f, 0.f, 0.f),
      P_(cv::Matx44f::zeros()),
      H_(1.f, 0.f, 0.f, 0.f,
         0.f, 1.f, 0.f, 0.f),
      R_(R), sigmaA_(sigmaA), sigmaV0_(sigmaV0)
{}

void Tracker::init(cv::Point2f z) {
    x_ = cv::Vec4f(z.x, z.y, 0.f, 0.f);
    P_ = cv::Matx44f::zeros();
    P_(0, 0) = R_(0, 0);
    P_(1, 1) = R_(1, 1);
    P_(2, 2) = sigmaV0_ * sigmaV0_;
    P_(3, 3) = sigmaV0_ * sigmaV0_;
    y_ = cv::Vec2f(0.f, 0.f);
    initialized_ = true;
}

void Tracker::reset() {
    initialized_ = false;
    missed_ = 0;
}

void Tracker::predict(float dt) {
    if (!initialized_) return;

    cv::Matx44f F(1.f, 0.f,  dt, 0.f,
                  0.f, 1.f, 0.f,  dt,
                  0.f, 0.f, 1.f, 0.f,
                  0.f, 0.f, 0.f, 1.f);

    const float sa2 = sigmaA_ * sigmaA_;
    const float t2  = dt * dt;
    const float t3  = t2 * dt;
    const float t4  = t3 * dt;
    const float q11 = sa2 * t4 * 0.25f;
    const float q13 = sa2 * t3 * 0.5f;
    const float q33 = sa2 * t2;

    cv::Matx44f Q(q11, 0.f, q13, 0.f,
                  0.f, q11, 0.f, q13,
                  q13, 0.f, q33, 0.f,
                  0.f, q13, 0.f, q33);

    x_ = F * x_;
    P_ = F * P_ * F.t() + Q;
}

void Tracker::update(cv::Point2f z) {
    if (!initialized_) { init(z); return; }

    cv::Vec2f zv(z.x, z.y);
    y_ = zv - H_ * x_;

    cv::Matx22f S = H_ * P_ * H_.t() + R_;
    cv::Matx<float, 4, 2> K = P_ * H_.t() * S.inv();

    x_ = x_ + K * y_;

    cv::Matx44f IKH = cv::Matx44f::eye() - K * H_;
    P_ = IKH * P_ * IKH.t() + K * R_ * K.t();
}
