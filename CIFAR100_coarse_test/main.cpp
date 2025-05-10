// main.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <stdexcept>
#include <cstring>  // memcpy를 위해 추가

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

// CIFAR-100 test.bin 파싱 - 코스 레이블을 사용하도록 수정
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
        // 여기서 fine 대신 coarse 레이블을 사용
        labels[i] = static_cast<int>(coarse);
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
        // 배치 크기 설정 (메모리 전송 오버헤드 감소를 위해)
        const int BATCH_SIZE = 1;  // 배치 크기를 64로 설정
        
        // 1) ONNX 모델 로드 및 파싱 (explicit batch)
        auto onnxData = readFile("resnet-101.onnx");
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

        // 2) BuilderConfig 및 Optimization Profile 설정
        IBuilderConfig* config = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(1ULL << 30);
        if (builder->platformHasFastFp16()) {
            config->setFlag(BuilderFlag::kFP16);
        }

        IOptimizationProfile* profile = builder->createOptimizationProfile();
        // 배치 크기 지정으로 변경
        profile->setDimensions("input", OptProfileSelector::kMIN, Dims4{1,3,32,32});
        profile->setDimensions("input", OptProfileSelector::kOPT, Dims4{BATCH_SIZE,3,32,32});
        profile->setDimensions("input", OptProfileSelector::kMAX, Dims4{BATCH_SIZE,3,32,32});
        config->addOptimizationProfile(profile);

        // 3) 엔진 빌드 및 컨텍스트 생성
        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
        if (!engine) {
            std::cerr << "ERROR: Engine build failed\n";
            return -1;
        }
        IExecutionContext* context = engine->createExecutionContext();

        // 4) 바인딩 인덱스 조회 및 차원 설정
        int inputIdx  = engine->getBindingIndex("input");
        int outputIdx = engine->getBindingIndex("output");
        context->setBindingDimensions(inputIdx, Dims4{BATCH_SIZE,3,32,32});
        auto inDims  = context->getBindingDimensions(inputIdx);
        auto outDims = context->getBindingDimensions(outputIdx);

        // 5) 버퍼 크기 계산 및 할당 (배치 처리를 위한 크기 조정)
        size_t inputSizePerImage = 3 * 32 * 32 * sizeof(float);
        size_t inputSize = BATCH_SIZE * inputSizePerImage;
        
        size_t outputSizePerImage = sizeof(float);
        for (int i = 1; i < outDims.nbDims; ++i) outputSizePerImage *= outDims.d[i]; // 첫 번째 차원(배치)은 제외
        size_t outputSize = BATCH_SIZE * outputSizePerImage;

        // CUDA 스트림 생성
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // 핀드 메모리 할당 (빠른 메모리 전송을 위해) - 타입캐스팅 수정
        void* h_input_void = nullptr;
        void* h_output_void = nullptr;
        cudaMallocHost(&h_input_void, inputSize);
        cudaMallocHost(&h_output_void, outputSize);
        
        // 타입 캐스팅
        float* h_input = static_cast<float*>(h_input_void);
        float* h_output = static_cast<float*>(h_output_void);

        // GPU 메모리 할당
        void* dBuf[2] = { nullptr, nullptr };
        cudaMalloc(&dBuf[inputIdx], inputSize);
        cudaMalloc(&dBuf[outputIdx], outputSize);

        // 6) CIFAR-100 테스트 데이터 로드
        std::vector<std::vector<float>> images;
        std::vector<int> labels;
        if (!loadCIFAR100Test("test.bin", images, labels)) {
            std::cerr << "Failed to load CIFAR-100 test data\n";
            return -1;
        }

        // CIFAR-100 코스 레이블 (20개)에 대한 매핑 출력
        std::cout << "Using CIFAR-100 coarse labels (20 classes) instead of fine labels (100 classes)\n";
        std::cout << "Coarse labels represent broader categories like vehicles, animals, etc.\n\n";

        // 7) 배치 처리를 통한 추론 및 정확도 계산 (Top-1, Top-5) + 타이밍
        int top1_correct = 0, top5_correct = 0;
        const int total = static_cast<int>(images.size());
        auto t_start = std::chrono::high_resolution_clock::now();
        
        // 파인 레이블에서 코스 레이블로의 매핑을 만들어야 함
        std::vector<int> fine_to_coarse(100, -1);
        
        for (int batch_start = 0; batch_start < total; batch_start += BATCH_SIZE) {
            // 이 배치에서의 실제 이미지 수 계산
            int current_batch_size = std::min(BATCH_SIZE, total - batch_start);
            
            // 배치 입력 데이터 준비
            for (int i = 0; i < current_batch_size; ++i) {
                int img_idx = batch_start + i;
                memcpy(h_input + i * 3 * 32 * 32, images[img_idx].data(), inputSizePerImage);
            }
            
            // 배치 데이터를 한 번에 GPU로 전송
            cudaMemcpyAsync(dBuf[inputIdx], h_input, current_batch_size * inputSizePerImage, 
                           cudaMemcpyHostToDevice, stream);
            
            // 배치 실행 컨텍스트 설정
            if (current_batch_size != BATCH_SIZE) {
                context->setBindingDimensions(inputIdx, Dims4{current_batch_size, 3, 32, 32});
            }
            
            // 비동기 추론 실행
            context->enqueueV2(dBuf, stream, nullptr);
            
            // 결과를 한 번에 CPU로 복사
            cudaMemcpyAsync(h_output, dBuf[outputIdx], current_batch_size * outputSizePerImage,
                           cudaMemcpyDeviceToHost, stream);
            
            // 스트림 동기화 (결과가 완전히 전송될 때까지 대기)
            cudaStreamSynchronize(stream);
            
            // 배치 결과 처리
            for (int i = 0; i < current_batch_size; ++i) {
                int img_idx = batch_start + i;
                float* current_output = h_output + i * (outputSizePerImage / sizeof(float));
                int num_classes = outputSizePerImage / sizeof(float);
                
                // Top-1
                int pred1 = std::distance(current_output,
                                         std::max_element(current_output, current_output + num_classes));
                if (pred1 == labels[img_idx]) ++top1_correct;
                
                // Top-5
                std::vector<int> idxs(num_classes);
                std::iota(idxs.begin(), idxs.end(), 0);
                std::partial_sort(idxs.begin(), idxs.begin()+5, idxs.end(),
                                 [&](int a, int b){ return current_output[a] > current_output[b]; });
                for (int k = 0; k < 5; ++k) {
                    if (idxs[k] == labels[img_idx]) { ++top5_correct; break; }
                }
            }
            
            if ((batch_start + current_batch_size) % 1000 == 0 || (batch_start + current_batch_size) == total) {
                std::cout << "Processed " << (batch_start + current_batch_size) << " / " << total 
                          << "\r" << std::flush;
            }
        }
        
        auto t_end = std::chrono::high_resolution_clock::now();
        auto inf_time = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        std::cout << "\n**Coarse Classification Results:**\n";
        std::cout << "**Top-1 Accuracy:** " << (100.0 * top1_correct / total) << "%\n";
        std::cout << "**Top-5 Accuracy:** " << (100.0 * top5_correct / total) << "%\n";
        std::cout << "**Total inference time:** " << inf_time << " ms\n";
        std::cout << "**Average time per image:** " << (inf_time / total) << " ms\n";

        // 8) 리소스 해제
        cudaStreamDestroy(stream);
        cudaFreeHost(h_input_void);
        cudaFreeHost(h_output_void);
        cudaFree(dBuf[inputIdx]);
        cudaFree(dBuf[outputIdx]);
        context->destroy();
        engine->destroy();
        config->destroy();
        parser->destroy();
        network->destroy();
        builder->destroy();
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}