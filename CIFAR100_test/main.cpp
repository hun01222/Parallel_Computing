#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <stdexcept>
#include <cstring>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;

std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open " + filename);
    size_t size = file.tellg();
    std::vector<char> buf(size);
    file.seekg(0);
    file.read(buf.data(), size);
    return buf;
}

void saveEngine(ICudaEngine* engine, const std::string& filename) {
    IHostMemory* serialized = engine->serialize();
    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
    serialized->destroy();
}

ICudaEngine* buildEngineFromOnnx(const std::string& onnxPath, Logger& logger) {
    auto onnxData = readFile(onnxPath);
    IBuilder* builder = createInferBuilder(logger);
    const uint32_t explicitBatch = 1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    auto parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parse(onnxData.data(), onnxData.size()))
        throw std::runtime_error("ONNX parse failed: " + onnxPath);

    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1ULL << 30);
    if (builder->platformHasFastFp16()) config->setFlag(BuilderFlag::kFP16);

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();

    if (!engine) throw std::runtime_error("Engine build failed: " + onnxPath);
    return engine;
}

ICudaEngine* loadEngineFromFile(const std::string& filename, Logger& logger) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open engine file: " + filename);
    std::vector<char> buf((std::istreambuf_iterator<char>(ifs)), {});
    IRuntime* runtime = createInferRuntime(logger);
    return runtime->deserializeCudaEngine(buf.data(), buf.size(), nullptr);
}

void convertAllModels(Logger& logger) {
    std::vector<std::string> modelNames = {"coarse_classifier"};
    for (int i = 0; i < 20; ++i) modelNames.push_back("fine_" + std::to_string(i));

    for (const auto& name : modelNames) {
        std::string engineFile = name + ".engine";
        if (std::ifstream(engineFile)) continue; // 이미 변환되어 있으면 skip
        std::cout << "[Build] Converting " << name << "...\n";
        ICudaEngine* engine = buildEngineFromOnnx(name + ".onnx", logger);
        saveEngine(engine, engineFile);
        engine->destroy();
    }
}

bool loadCIFAR100Test(const std::string& filename, std::vector<std::vector<float>>& images, std::vector<int>& labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;
    const int N = 10000, IMG_SZ = 3 * 32 * 32;
    images.resize(N, std::vector<float>(IMG_SZ));
    labels.resize(N);
    const float mean[3] = {0.5071f, 0.4865f, 0.4409f};
    const float stdv[3] = {0.2673f, 0.2564f, 0.2762f};

    for (int i = 0; i < N; ++i) {
        unsigned char coarse, fine;
        file.read(reinterpret_cast<char*>(&coarse), 1);
        file.read(reinterpret_cast<char*>(&fine), 1);
        labels[i] = static_cast<int>(fine);
        std::vector<unsigned char> raw(IMG_SZ);
        file.read(reinterpret_cast<char*>(raw.data()), IMG_SZ);
        for (int c = 0; c < 3; ++c) {
            int offset = c * 32 * 32;
            for (int px = 0; px < 32 * 32; ++px) {
                float v = raw[offset + px] / 255.0f;
                images[i][offset + px] = (v - mean[c]) / stdv[c];
            }
        }
    }
    return true;
}

int main() {
    try {
        Logger gLogger;
        convertAllModels(gLogger);

        std::vector<std::vector<float>> images;
        std::vector<int> labels;
        if (!loadCIFAR100Test("test.bin", images, labels))
            throw std::runtime_error("Failed to load test.bin");

        ICudaEngine* coarseEngine = loadEngineFromFile("coarse_classifier.engine", gLogger);
        IExecutionContext* coarseContext = coarseEngine->createExecutionContext();

        std::vector<ICudaEngine*> fineEngines(20);
        std::vector<IExecutionContext*> fineContexts(20);
        for (int i = 0; i < 20; ++i) {
            fineEngines[i] = loadEngineFromFile("fine_" + std::to_string(i) + ".engine", gLogger);
            fineContexts[i] = fineEngines[i]->createExecutionContext();
        }

        const int IMG_SZ = 3 * 32 * 32;
        void* dInput = nullptr;
        void* dCoarseOut = nullptr;
        void* dFineOut = nullptr;
        cudaMalloc(&dInput, IMG_SZ * sizeof(float));
        cudaMalloc(&dCoarseOut, 20 * sizeof(float)); // coarse 20 class
        cudaMalloc(&dFineOut, 5 * sizeof(float));    // fine 5 class 예시

        // 수정된 호스트 메모리 할당 (reinterpret_cast 적용)
        float* hInput;
        float* hCoarseOut;
        float* hFineOut;
        cudaMallocHost(reinterpret_cast<void**>(&hInput),
                      IMG_SZ * sizeof(float));
        cudaMallocHost(reinterpret_cast<void**>(&hCoarseOut),
                      20 * sizeof(float));
        cudaMallocHost(reinterpret_cast<void**>(&hFineOut),
                      5 * sizeof(float));

        for (size_t i = 0; i < images.size(); ++i) {
            memcpy(hInput, images[i].data(), IMG_SZ * sizeof(float));
            cudaMemcpy(dInput, hInput, IMG_SZ * sizeof(float), cudaMemcpyHostToDevice);

            void* coarseBuf[] = {dInput, dCoarseOut};
            coarseContext->enqueueV2(coarseBuf, 0, nullptr);
            cudaMemcpy(hCoarseOut, dCoarseOut, 20 * sizeof(float), cudaMemcpyDeviceToHost);

            int coarseLabel = std::distance(hCoarseOut, std::max_element(hCoarseOut, hCoarseOut + 20));

            void* fineBuf[] = {dInput, dFineOut};
            fineContexts[coarseLabel]->enqueueV2(fineBuf, 0, nullptr);
            cudaMemcpy(hFineOut, dFineOut, 5 * sizeof(float), cudaMemcpyDeviceToHost);

            int fineLabel = std::distance(hFineOut, std::max_element(hFineOut, hFineOut + 5));

            std::cout << "[Image " << i << "] Coarse: " << coarseLabel << ", Fine: " << fineLabel << "\n";
        }

        for (auto ctx : fineContexts) ctx->destroy();
        for (auto eng : fineEngines) eng->destroy();
        coarseContext->destroy();
        coarseEngine->destroy();

        cudaFree(dInput);
        cudaFree(dCoarseOut);
        cudaFree(dFineOut);
        cudaFreeHost(hInput);
        cudaFreeHost(hCoarseOut);
        cudaFreeHost(hFineOut);
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
