// main.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <stdexcept>
#include <cstring>  // memcpy

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

// Logger 구현
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;

// 파일 전체를 메모리에 로드
std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open " + filename);
    size_t size = file.tellg();
    std::vector<char> buf(size);
    file.seekg(0);
    file.read(buf.data(), size);
    return buf;
}

// 파일 존재 여부 확인
bool fileExists(const std::string& name) {
    std::ifstream f(name);
    return f.good();
}

// CIFAR-100 test.bin 파싱
bool loadCIFAR100Test(const std::string& filename,
                      std::vector<std::vector<float>>& images,
                      std::vector<int>& labels) {
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
        file.read(reinterpret_cast<char*>(&fine),   1);
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
        const int BATCH_SIZE = 1;
        const std::string onnxFile = "resnet-152.onnx";
        const std::string engineFile = "resnet-152.trt";

        // TensorRT 엔진 생성 또는 로드
        ICudaEngine* engine = nullptr;
        if (fileExists(engineFile)) {
            std::cout << "Loading serialized engine from " << engineFile << std::endl;
            auto trtData = readFile(engineFile);
            IRuntime* runtime = createInferRuntime(gLogger);
            engine = runtime->deserializeCudaEngine(trtData.data(), trtData.size(), nullptr);
            runtime->destroy();
            if (!engine) throw std::runtime_error("Failed to deserialize engine");
        } else {
            std::cout << "Building engine from ONNX: " << onnxFile << std::endl;
            auto onnxData = readFile(onnxFile);
            IBuilder* builder = createInferBuilder(gLogger);
            const uint32_t explicitBatch = 1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
            INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
            auto parser = nvonnxparser::createParser(*network, gLogger);
            if (!parser->parse(onnxData.data(), onnxData.size())) {
                std::cerr << "ERROR: ONNX parse failed\n";
                for (int i = 0; i < parser->getNbErrors(); ++i) {
                    auto err = parser->getError(i);
                    std::cerr << "  [" << i << "] " << err->desc()
                              << " (node " << err->node() << ")\n";
                }
                return -1;
            }

            IBuilderConfig* config = builder->createBuilderConfig();
            config->setMaxWorkspaceSize(1ULL << 30);
            if (builder->platformHasFastFp16()) config->setFlag(BuilderFlag::kFP16);

            IOptimizationProfile* profile = builder->createOptimizationProfile();
            profile->setDimensions("input", OptProfileSelector::kMIN, Dims4{1,3,32,32});
            profile->setDimensions("input", OptProfileSelector::kOPT, Dims4{BATCH_SIZE,3,32,32});
            profile->setDimensions("input", OptProfileSelector::kMAX, Dims4{BATCH_SIZE,3,32,32});
            config->addOptimizationProfile(profile);

            engine = builder->buildEngineWithConfig(*network, *config);
            if (!engine) throw std::runtime_error("Engine build failed");

            // Serialize engine to file
            IHostMemory* serialized = engine->serialize();
            std::ofstream out(engineFile, std::ios::binary);
            out.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
            serialized->destroy();

            // 리소스 해제
            parser->destroy();
            network->destroy();
            config->destroy();
            builder->destroy();
        }

        // 실행 컨텍스트 생성
        IExecutionContext* context = engine->createExecutionContext();

        // 바인딩 설정
        int inputIdx  = engine->getBindingIndex("input");
        int outputIdx = engine->getBindingIndex("output");
        context->setBindingDimensions(inputIdx, Dims4{BATCH_SIZE,3,32,32});
        auto inDims  = context->getBindingDimensions(inputIdx);
        auto outDims = context->getBindingDimensions(outputIdx);

        size_t inputSizePerImage = 3 * 32 * 32 * sizeof(float);
        size_t inputSize = BATCH_SIZE * inputSizePerImage;

        size_t outputSizePerImage = sizeof(float);
        for (int i = 1; i < outDims.nbDims; ++i)
            outputSizePerImage *= outDims.d[i];
        size_t outputSize = BATCH_SIZE * outputSizePerImage;

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        void* h_input_void  = nullptr;
        void* h_output_void = nullptr;
        cudaMallocHost(&h_input_void, inputSize);
        cudaMallocHost(&h_output_void, outputSize);
        float* h_input  = static_cast<float*>(h_input_void);
        float* h_output = static_cast<float*>(h_output_void);

        void* dBuf[2] = { nullptr, nullptr };
        cudaMalloc(&dBuf[inputIdx], inputSize);
        cudaMalloc(&dBuf[outputIdx], outputSize);

        // CIFAR-100 데이터 로드
        std::vector<std::vector<float>> images;
        std::vector<int> labels;
        if (!loadCIFAR100Test("test.bin", images, labels)) {
            std::cerr << "Failed to load CIFAR-100 test data\n";
            return -1;
        }

        // 추론 및 정확도 계산
        int top1 = 0, top5 = 0;
        int total = static_cast<int>(images.size());
        auto t_start = std::chrono::high_resolution_clock::now();

        for (int bs = 0; bs < total; bs += BATCH_SIZE) {
            int cur_bs = std::min(BATCH_SIZE, total - bs);
            for (int i = 0; i < cur_bs; ++i)
                memcpy(h_input + i * 3 * 32 * 32,
                       images[bs + i].data(), inputSizePerImage);

            cudaMemcpyAsync(dBuf[inputIdx], h_input,
                            cur_bs * inputSizePerImage,
                            cudaMemcpyHostToDevice, stream);
            if (cur_bs != BATCH_SIZE)
                context->setBindingDimensions(inputIdx, Dims4{cur_bs,3,32,32});

            context->enqueueV2(dBuf, stream, nullptr);
            cudaMemcpyAsync(h_output, dBuf[outputIdx],
                            cur_bs * outputSizePerImage,
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            for (int i = 0; i < cur_bs; ++i) {
                float* out_ptr = h_output + i * (outputSizePerImage/sizeof(float));
                int classes = outputSizePerImage/sizeof(float);
                int p1 = std::distance(out_ptr,
                                      std::max_element(out_ptr, out_ptr+classes));
                if (p1 == labels[bs+i]) ++top1;
                std::vector<int> idxs(classes);
                std::iota(idxs.begin(), idxs.end(), 0);
                std::partial_sort(idxs.begin(), idxs.begin()+5, idxs.end(),
                                  [&](int a,int b){return out_ptr[a]>out_ptr[b];});
                for (int k=0; k<5; ++k) if (idxs[k]==labels[bs+i]) {++top5; break;}
            }
            if ((bs + cur_bs) % 1000 == 0 || (bs+cur_bs)==total)
                std::cout << "Processed " << (bs+cur_bs) << " / " << total << "\r";
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double inf_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        std::cout << "\n**Top-1 Accuracy:** " << (100. * top1 / total) << "%\n";
        std::cout << "**Top-5 Accuracy:** " << (100. * top5 / total) << "%\n";
        std::cout << "**Total inference time:** " << inf_ms << " ms\n";
        std::cout << "**Average time per image:** " << (inf_ms / total) << " ms\n";

        // 리소스 해제
        cudaStreamDestroy(stream);
        cudaFreeHost(h_input_void);
        cudaFreeHost(h_output_void);
        cudaFree(dBuf[inputIdx]);
        cudaFree(dBuf[outputIdx]);
        context->destroy();
        engine->destroy();
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
