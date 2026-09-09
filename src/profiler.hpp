// profiler.hpp — header-only latency instrumentation for the Iron-Dome loop.
//
// C++17, no dependencies, no build system changes. Drop the file in your
// include path and #include it.
//
//     #include "profiler.hpp"
//
//     static prof::Profiler P(30);            // discard 30 warmup frames
//
//     int main() {
//         P.install_signal_handler();         // Ctrl-C still prints the report
//
//         while (running) {
//             PROF_FRAME(P);                       // whole iteration
//             { PROF_STAGE(P, "capture");   grab_frame(); }
//             { PROF_STAGE(P, "inference"); run_detector(); }
//             { PROF_STAGE(P, "kalman");    kf.update(); }
//             { PROF_STAGE(P, "serial");    write_command(); }
//         }
//         P.report();
//     }
//
// The macros are RAII scope guards: the timer starts where the macro sits and
// stops when the enclosing { } block closes. That is what "wrap a stage"
// means here — put the macro inside a block whose braces bound the work you
// want timed. PROF_FRAME goes at the top of the loop body, so it spans one
// full iteration.
//
// CUDA: kernels launch asynchronously, so timing an inference call without
// synchronizing measures how long it took to *queue* the work, not to run it.
// Register a sync callback once and use PROF_STAGE_SYNC on GPU stages:
//
//     P.set_sync([]{ cudaDeviceSynchronize(); });
//     { PROF_STAGE_SYNC(P, "inference"); run_detector(); }
//
// The header stays CUDA-free, so it compiles whether or not you link CUDA.

#ifndef IRON_DOME_PROFILER_HPP
#define IRON_DOME_PROFILER_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace prof {

using Clock = std::chrono::steady_clock;

struct Stats {
    std::size_t n = 0;
    double mean = 0, stdev = 0, min = 0, p50 = 0, p95 = 0, p99 = 0, max = 0;
};

class Samples {
public:
    void add(double ms) { v_.push_back(ms); }
    std::size_t size() const { return v_.size(); }

    Stats stats() const {
        Stats s;
        if (v_.empty()) return s;
        std::vector<double> a = v_;
        std::sort(a.begin(), a.end());
        s.n = a.size();
        double sum = 0;
        for (double x : a) sum += x;
        s.mean = sum / a.size();
        double acc = 0;
        for (double x : a) acc += (x - s.mean) * (x - s.mean);
        s.stdev = a.size() > 1 ? std::sqrt(acc / a.size()) : 0.0;
        s.min = a.front();
        s.max = a.back();
        s.p50 = pct(a, 0.50);
        s.p95 = pct(a, 0.95);
        s.p99 = pct(a, 0.99);
        return s;
    }

private:
    static double pct(const std::vector<double>& a, double q) {
        // nearest-rank
        long n = static_cast<long>(a.size());
        long k = static_cast<long>(std::lround(q * n + 0.5)) - 1;
        k = std::max(0L, std::min(n - 1, k));
        return a[static_cast<std::size_t>(k)];
    }
    std::vector<double> v_;
};

class Profiler {
public:
    explicit Profiler(std::size_t warmup = 30,
                      std::string label = "Iron-Dome pipeline")
        : warmup_(warmup), label_(std::move(label)) {}

    void set_sync(std::function<void()> fn) { sync_ = std::move(fn); }
    void sync() const { if (sync_) sync_(); }
    bool warm() const { return frames_ > warmup_; }

    void add(const std::string& name, double ms) {
        if (!warm()) return;
        auto it = stages_.find(name);
        if (it == stages_.end()) {
            order_.push_back(name);
            it = stages_.emplace(name, Samples{}).first;
        }
        it->second.add(ms);
    }

    void begin_frame() { ++frames_; }

    void end_frame(double ms, Clock::time_point t0, Clock::time_point t1) {
        if (!warm()) return;
        e2e_.add(ms);
        ++measured_;
        if (!have_first_) { first_ = t0; have_first_ = true; }
        last_ = t1;
    }

    double fps() const {
        if (!have_first_ || measured_ == 0) return 0.0;
        double span = std::chrono::duration<double>(last_ - first_).count();
        return span > 0 ? measured_ / span : 0.0;
    }

    // ---- output ----------------------------------------------------------
    void report(std::ostream& os = std::cout) {
        if (reported_) return;
        reported_ = true;

        if (measured_ == 0) {
            os << "\n[" << label_ << "] no measured frames (" << frames_
               << " seen, warmup=" << warmup_ << ").\n";
            return;
        }

        const int W = 11;
        std::ostringstream hdr;
        hdr << std::left << std::setw(16) << "stage" << std::right
            << std::setw(8) << "n" << std::setw(W) << "mean"
            << std::setw(W) << "p50" << std::setw(W) << "p95"
            << std::setw(W) << "p99" << std::setw(W) << "max"
            << std::setw(W) << "stdev";
        std::string bar(hdr.str().size(), '=');
        std::string dash(hdr.str().size(), '-');

        os << "\n" << bar << "\n"
           << label_ << " — latency (ms), " << measured_ << " frames after "
           << std::min(frames_, warmup_) << " warmup\n"
           << bar << "\n" << hdr.str() << "\n" << dash << "\n";

        double stage_p50_sum = 0;
        for (const auto& name : order_) {
            Stats s = stages_.at(name).stats();
            if (!s.n) continue;
            stage_p50_sum += s.p50;
            row(os, name, s, W);
        }

        Stats e = e2e_.stats();
        os << dash << "\n";
        row(os, "END-TO-END", e, W);
        os << bar << "\n";
        os << "sustained throughput: " << std::fixed << std::setprecision(1)
           << fps() << " FPS\n";

        if (!order_.empty() && e.p50 > 0) {
            double gap = e.p50 - stage_p50_sum;
            os << "unaccounted (p50):    " << std::showpos
               << std::setprecision(2) << gap << std::noshowpos << " ms ("
               << std::setprecision(1) << (gap / e.p50 * 100.0) << "% of loop)\n";
            if (gap > 0.25 * e.p50)
                os << "  ^ over a quarter of the loop is outside your "
                      "instrumented stages.\n";
        }

        // dominant stage
        const std::string* worst = nullptr;
        double worst_p50 = -1;
        for (const auto& name : order_) {
            Stats s = stages_.at(name).stats();
            if (s.n && s.p50 > worst_p50) { worst_p50 = s.p50; worst = &name; }
        }

        os << "\n--- numbers you can defend in an interview ---\n"
           << "  sustained rate:      " << std::setprecision(0) << fps()
           << " FPS\n"
           << "  end-to-end latency:  " << std::setprecision(0) << e.p50
           << " ms median, " << e.p95 << " ms p95\n"
           << "  loop jitter (stdev): " << std::setprecision(1) << e.stdev
           << " ms\n";
        if (worst)
            os << "  dominant stage:      " << *worst << " ("
               << std::setprecision(0) << worst_p50 << " ms, "
               << (worst_p50 / e.p50 * 100.0) << "% of the loop)\n";
        os << "  Quote median and p95 together. A mean alone hides the tail,\n"
              "  and the tail is what makes a tracker lose lock.\n";
    }

    void to_json(const std::string& path) {
        std::ofstream f(path);
        if (!f) return;
        Stats e = e2e_.stats();
        f << "{\n  \"label\": \"" << label_ << "\",\n"
          << "  \"frames_total\": " << frames_ << ",\n"
          << "  \"frames_measured\": " << measured_ << ",\n"
          << "  \"sustained_fps\": " << fps() << ",\n"
          << "  \"end_to_end_ms\": " << json(e) << ",\n"
          << "  \"stages_ms\": {\n";
        for (std::size_t i = 0; i < order_.size(); ++i) {
            f << "    \"" << order_[i] << "\": "
              << json(stages_.at(order_[i]).stats())
              << (i + 1 < order_.size() ? "," : "") << "\n";
        }
        f << "  }\n}\n";
    }

    void install_signal_handler() {
        instance_ = this;
        std::signal(SIGINT, &Profiler::on_signal);
        std::signal(SIGTERM, &Profiler::on_signal);
    }

private:
    static void row(std::ostream& os, const std::string& name,
                    const Stats& s, int W) {
        os << std::left << std::setw(16) << name << std::right
           << std::setw(8) << s.n << std::fixed << std::setprecision(2)
           << std::setw(W) << s.mean << std::setw(W) << s.p50
           << std::setw(W) << s.p95 << std::setw(W) << s.p99
           << std::setw(W) << s.max << std::setw(W) << s.stdev << "\n";
    }

    static std::string json(const Stats& s) {
        std::ostringstream o;
        o << std::fixed << std::setprecision(3)
          << "{\"n\": " << s.n << ", \"mean\": " << s.mean
          << ", \"stdev\": " << s.stdev << ", \"min\": " << s.min
          << ", \"p50\": " << s.p50 << ", \"p95\": " << s.p95
          << ", \"p99\": " << s.p99 << ", \"max\": " << s.max << "}";
        return o.str();
    }

    static void on_signal(int) {
        if (instance_) instance_->report(std::cerr);
        std::_Exit(0);
    }

    static inline Profiler* instance_ = nullptr;

    std::size_t warmup_, frames_ = 0, measured_ = 0;
    std::string label_;
    std::vector<std::string> order_;
    std::unordered_map<std::string, Samples> stages_;
    Samples e2e_;
    Clock::time_point first_, last_;
    bool have_first_ = false, reported_ = false;
    std::function<void()> sync_;
};

// ---- RAII scope guards ---------------------------------------------------
class StageGuard {
public:
    StageGuard(Profiler& p, const char* name, bool do_sync)
        : p_(p), name_(name), sync_(do_sync), t0_(Clock::now()) {}
    ~StageGuard() {
        if (sync_) p_.sync();
        double ms = std::chrono::duration<double, std::milli>(
                        Clock::now() - t0_).count();
        p_.add(name_, ms);
    }
private:
    Profiler& p_;
    const char* name_;
    bool sync_;
    Clock::time_point t0_;
};

class FrameGuard {
public:
    explicit FrameGuard(Profiler& p) : p_(p), t0_(Clock::now()) {
        p_.begin_frame();
    }
    ~FrameGuard() {
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0_).count();
        p_.end_frame(ms, t0_, t1);
    }
private:
    Profiler& p_;
    Clock::time_point t0_;
};

}  // namespace prof

#define PROF_CAT_(a, b) a##b
#define PROF_CAT(a, b) PROF_CAT_(a, b)
#define PROF_FRAME(P) ::prof::FrameGuard PROF_CAT(_pf_, __LINE__)(P)
#define PROF_STAGE(P, NAME) \
    ::prof::StageGuard PROF_CAT(_ps_, __LINE__)((P), (NAME), false)
#define PROF_STAGE_SYNC(P, NAME) \
    ::prof::StageGuard PROF_CAT(_ps_, __LINE__)((P), (NAME), true)

#endif  // IRON_DOME_PROFILER_HPP
