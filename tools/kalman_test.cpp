#include "Tracker.hpp"
#include <cmath>
#include <cstdio>
#include <random>
#include <cstdlib>

void test3(float sigmaA);

int main(int argc, char** argv){
	const float sigmaA = (argc > 1) ? std::atof(argv[1]) : 306.f;
	const float dt = 0.0333f;
	const int N = 200;
	const float sdX = 0.240f;
	const float sdY = 0.189f;

	const float x0 = 100.f, vx = 50.f;
	const float y0 = 200.f, vy = 30.f;

	std::mt19937 rng(42);
	std::normal_distribution<float> nx(0.f, sdX), ny(0.f, sdY);

	Tracker tracker(cv::Matx22f(sdX * sdX, 0.f, 0.f, sdY * sdY), sigmaA, 300.f);

	double sumErr = 0.0;
	double sumRaw = 0.0;
	int counted = 0;

	for(int k = 0; k < N; ++k){
		float t = k * dt;
		cv::Point2f truth(x0 + vx * t, y0 + vy * t);
		cv::Point2f z (truth.x + nx(rng), truth.y + ny(rng));

		if(k==0)
			tracker.init(z);
		else{
			tracker.predict(dt);
			tracker.update(z);
		}

		if(k >= 20){
			cv::Point2f est = tracker.position();
			sumErr += std::hypot(est.x - truth.x, est.y - truth.y);
			sumRaw += std::hypot(z.x - truth.x, z.y - truth.y);
			++counted;
		}
	}

	cv::Point2f v = tracker.velocity();
	std::printf("Test 1 (const velocity)\n");
	std::printf("   mean |pos err| : %.3f px  (noise sd %.3f, %.3f)\n",
			sumErr / counted, sdX, sdY);
	std::printf("   mean |raw err| : %.3f px\n", sumRaw/counted);
	std::printf("   final velocity : %.1f, %.1f    (truth %.1f, %.1f)\n",
			v.x, v.y, vx, vy);
	test3(sigmaA);
	return 0;
}

void test3(float sigmaA) {
    const float dt = 0.0333f, sdX = 0.240f, sdY = 0.189f;
    const int N = 300, flip = 150;

    std::mt19937 rng(42);
    std::normal_distribution<float> nx(0.f, sdX), ny(0.f, sdY);
    Tracker tracker(cv::Matx22f(sdX*sdX, 0.f, 0.f, sdY*sdY), sigmaA, 300.f);

    float tx = 100.f, ty = 200.f;
    int   recovery = -1;

    for (int k = 0; k < N; ++k) {
        float vx = (k < flip) ? 200.f : -200.f;
        if (k > 0) { tx += vx * dt; ty += 30.f * dt; }

        cv::Point2f z(tx + nx(rng), ty + ny(rng));
        if (k == 0) tracker.init(z);
        else { tracker.predict(dt); tracker.update(z); }

        if (k > flip && recovery < 0) {
            cv::Point2f e = tracker.position();
            if (std::hypot(e.x - tx, e.y - ty) < 1.0f) recovery = k - flip;
        }
    }
    std::printf("Test 3 sigma_a=%-5.0f recovery: %d frames (%.0f ms)\n",
                sigmaA, recovery, recovery * dt * 1000.f);
}
