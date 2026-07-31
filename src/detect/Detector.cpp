#include "Detector.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "NvInfer.h"
#include "preprocess/Preprocessor.hpp"

using namespace nvinfer1;

namespace {

class Logger : public ILogger
{
    void log(Severity severity, char const* msg) noexcept override
    {
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
};

std::vector<char> loadEngine(std::string const& enginePath)
{
    std::ifstream f(enginePath, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open engine: " + enginePath);
    std::streamsize const size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<size_t>(size));
    if (!f.read(bytes.data(), size)) throw std::runtime_error("short read on engine");
    return bytes;
}

size_t volume(Dims const& d)
{
    size_t v = 1;
    for (int32_t i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
    return v;
}

} // namespace

struct Detector::Impl {
    Logger logger;                                  // must outlive all TRT objects
    std::unique_ptr<IRuntime>          runtime;
    std::unique_ptr<ICudaEngine>       engine;
    std::unique_ptr<IExecutionContext> context;
    cudaStream_t stream{nullptr};
    char const*  inputName{nullptr};
    char const*  outputName{nullptr};
    void*        dInput{nullptr};
    void*        dOutput{nullptr};
    size_t       inputCount{0};
    std::vector<float> hostOutput;
    float confThreshold{0.20f};

    ~Impl() {
        if (dInput)  cudaFree(dInput);
        if (dOutput) cudaFree(dOutput);
        if (stream)  cudaStreamDestroy(stream);
    }
};

Detector::Detector(std::string const& enginePath, float confThreshold)
    : impl_(std::make_unique<Impl>())
{
    auto& d = *impl_;
    d.confThreshold = confThreshold;

    auto engineBytes = loadEngine(enginePath);
    d.runtime.reset(createInferRuntime(d.logger));
    d.engine.reset(d.runtime->deserializeCudaEngine(engineBytes.data(), engineBytes.size()));
    if (!d.engine) throw std::runtime_error("deserializeCudaEngine failed");
    d.context.reset(d.engine->createExecutionContext());
    if (!d.context) throw std::runtime_error("createExecutionContext failed");

    d.inputName  = d.engine->getIOTensorName(0);
    d.outputName = d.engine->getIOTensorName(1);
    d.inputCount = volume(d.engine->getTensorShape(d.inputName));
    size_t const outputCount = volume(d.context->getTensorShape(d.outputName));
    d.hostOutput.resize(outputCount);

    cudaMalloc(&d.dInput,  d.inputCount * sizeof(float));
    cudaMalloc(&d.dOutput, outputCount  * sizeof(float));
    cudaStreamCreate(&d.stream);

    d.context->setTensorAddress(d.inputName,  d.dInput);
    d.context->setTensorAddress(d.outputName, d.dOutput);
}

Detector::~Detector() = default;
Detector::Detector(Detector&&) noexcept = default;
Detector& Detector::operator=(Detector&&) noexcept = default;

std::vector<Detection> Detector::detect(cv::Mat const& frame)
{
    auto& d = *impl_;
    Preprocessed pre = preprocess(frame);

    cudaMemcpyAsync(d.dInput, pre.blob.ptr<float>(),
                    d.inputCount * sizeof(float),
                    cudaMemcpyHostToDevice, d.stream);

    if (!d.context->enqueueV3(d.stream)) throw std::runtime_error("enqueueV3 failed");

    cudaMemcpyAsync(d.hostOutput.data(), d.dOutput,
                    d.hostOutput.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, d.stream);
    cudaStreamSynchronize(d.stream);

    std::vector<Detection> out;
    size_t const numRows = d.hostOutput.size() / 6;
    for (size_t i = 0; i < numRows; ++i) {
        float const* row = d.hostOutput.data() + i * 6;
        if (row[4] < d.confThreshold) break;
        out.push_back({
            (row[0] - pre.pad_left) / pre.scale,
            (row[1] - pre.pad_top)  / pre.scale,
            (row[2] - pre.pad_left) / pre.scale,
            (row[3] - pre.pad_top)  / pre.scale,
            row[4],
            static_cast<int>(row[5])
        });
    }
    return out;
}
