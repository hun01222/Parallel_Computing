// main.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <stdexcept>
#include <cstring>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

// nlohmann/json 헤더 (https://github.com/nlohmann/json)
#include <nlohmann/json.hpp>

using namespace nvinfer1;

// Logger for TensorRT
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;

// Read entire file into buffer
std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open " + filename);
    size_t size = file.tellg();
    std::vector<char> buf(size);
    file.seekg(0);
    file.read(buf.data(), size);
    return buf;
}

// Serialize and save engine
void saveEngine(ICudaEngine* engine, const std::string& filename) {
    IHostMemory* ser = engine->serialize();
    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(ser->data()), ser->size());
    ser->destroy();
}

// Build engine from ONNX
ICudaEngine* buildEngine(const std::string& onnxPath, Logger& logger) {
    auto data = readFile(onnxPath);
    IBuilder* builder = createInferBuilder(logger);
    uint32_t flags = 1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* net = builder->createNetworkV2(flags);
    auto parser = nvonnxparser::createParser(*net, logger);
    if (!parser->parse(data.data(), data.size()))
        throw std::runtime_error("ONNX parse failed: " + onnxPath);
    IBuilderConfig* cfg = builder->createBuilderConfig();
    cfg->setMaxWorkspaceSize(1ULL<<30);
    if (builder->platformHasFastFp16()) cfg->setFlag(BuilderFlag::kFP16);
    // dynamic batch
    auto input = net->getInput(0);
    if (input->getDimensions().d[0] == -1) {
        auto prof = builder->createOptimizationProfile();
        Dims dims = input->getDimensions();
        Dims min = dims, opt = dims, max = dims;
        min.d[0]=1; opt.d[0]=64; max.d[0]=64;
        prof->setDimensions(input->getName(), OptProfileSelector::kMIN, min);
        prof->setDimensions(input->getName(), OptProfileSelector::kOPT, opt);
        prof->setDimensions(input->getName(), OptProfileSelector::kMAX, max);
        cfg->addOptimizationProfile(prof);
    }
    ICudaEngine* engine = builder->buildEngineWithConfig(*net, *cfg);
    parser->destroy(); net->destroy(); cfg->destroy(); builder->destroy();
    if (!engine) throw std::runtime_error("Engine build failed");
    return engine;
}

// Load serialized engine
ICudaEngine* loadEngine(const std::string& file, Logger& logger) {
    std::ifstream ifs(file, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open engine: " + file);
    std::vector<char> buf((std::istreambuf_iterator<char>(ifs)), {});
    IRuntime* rt = createInferRuntime(logger);
    return rt->deserializeCudaEngine(buf.data(), buf.size(), nullptr);
}

// Convert ONNX models
void convertAll(Logger& logger) {
    std::vector<std::string> names={"resnet-101"};
    for(int i=0;i<20;i++) names.push_back("resnet-18_"+std::to_string(i));
    for(auto& n:names) {
        std::string eng=n+".engine";
        if(std::ifstream(eng)) continue;
        std::cout<<"Building "<<n<<"...\n";
        auto e=buildEngine(n+".onnx", logger);
        saveEngine(e, eng);
        e->destroy();
    }
}

// Load CIFAR-100 test
bool loadCIFAR100(const std::string& f, std::vector<std::vector<float>>& imgs,
                  std::vector<int>& cl, std::vector<int>& fl) {
    std::ifstream in(f, std::ios::binary);
    if(!in) return false;
    const int N=10000, S=3*32*32;
    imgs.resize(N, std::vector<float>(S));
    cl.resize(N); fl.resize(N);
    float mean[3]={0.5071f,0.4867f,0.4408f}, stdv[3]={0.2675f,0.2565f,0.2761f};
    for(int i=0;i<N;i++){
        unsigned char c,fine;
        in.read((char*)&c,1); in.read((char*)&fine,1);
        cl[i]=c; fl[i]=fine;
        std::vector<unsigned char> raw(S);
        in.read((char*)raw.data(), S);
        for(int ch=0;ch<3;ch++){
            int off=ch*32*32;
            for(int px=0;px<32*32;px++){
                float v=raw[off+px]/255.0f;
                imgs[i][off+px]=(v-mean[ch])/stdv[ch];
            }
        }
    }
    return true;
}

int main(){
    try{
        // 1) JSON 매핑 로드 ---------------------------------------------------
        std::ifstream ifs("coarse_to_fine.json");
        if (!ifs.is_open())
            throw std::runtime_error("Cannot open coarse_to_fine.json");
        nlohmann::json js; 
        ifs >> js;

        // 2) vector<array<int,5>> 형태로 변환
        std::vector<std::array<int,5>> coarse2fine(20);
        for (auto& item : js.items()) {
            int c = std::stoi(item.key());
            for (int i = 0; i < 5; ++i)
                coarse2fine[c][i] = item.value()[i].get<int>();
        }
        // ---------------------------------------------------------------------

        convertAll(gLogger);

        std::vector<std::vector<float>> images;
        std::vector<int> coarseLabels, fineLabels;
        if(!loadCIFAR100("test.bin", images, coarseLabels, fineLabels))
            throw std::runtime_error("Load failed");

        auto coarseE=loadEngine("resnet-101.engine",gLogger);
        auto coarseC=coarseE->createExecutionContext();
        std::vector<ICudaEngine*> fineE(20);
        std::vector<IExecutionContext*> fineC(20);
        for(int i=0;i<20;i++){
            fineE[i]=loadEngine("resnet-18_"+std::to_string(i)+".engine",gLogger);
            fineC[i]=fineE[i]->createExecutionContext();
        }

        const int S=3*32*32;
        void *dIn,*dC,*dF;
        cudaMalloc(&dIn, S*sizeof(float));
        cudaMalloc(&dC, 20*sizeof(float));
        cudaMalloc(&dF, 5*sizeof(float));

        float *hIn,*hC,*hF;
        cudaMallocHost((void**)&hIn, S*sizeof(float));
        cudaMallocHost((void**)&hC, 20*sizeof(float));
        cudaMallocHost((void**)&hF, 5*sizeof(float));

        int idxCo=coarseE->getBindingIndex(coarseE->getBindingName(0));
        Dims4 dCo{1,3,32,32}; coarseC->setBindingDimensions(idxCo,dCo);

        std::vector<int> idxFi(20);
        for(int i=0;i<20;i++)
            idxFi[i] = fineE[i]->getBindingIndex(fineE[i]->getBindingName(0));
        
        size_t corrCo=0, corrFi=0, corrBoth=0, hit=0, preC=0;
        auto t0=std::chrono::high_resolution_clock::now();


        
        std::vector<std::vector<float>> fineResults(20, std::vector<float>(5));  // coarse 모델에 대해 fine 모델 결과를 미리 저장할 공간
        std::vector<cudaStream_t> streams(2); // 2개의 stream 생성 (coarse, fine 모델)
        
        for(size_t i = 0; i < images.size(); i++) {
            // CUDA Stream을 사용하여 비동기적으로 처리
            cudaStreamCreate(&streams[0]);  // coarse stream
            cudaStreamCreate(&streams[1]);  // fine stream
        
            // 실제 배치 크기 설정 & 복사
            coarseC->setBindingDimensions(idxCo, Dims4{1,3,32,32});
            memcpy(hIn, images[i].data(), S * sizeof(float));
            cudaMemcpyAsync(dIn, hIn, S * sizeof(float), cudaMemcpyHostToDevice, streams[0]);
        
            // Coarse 추론 (비동기 처리)
            void* bufC[] = {dIn, dC};
            coarseC->enqueueV2(bufC, streams[0], nullptr);
            cudaMemcpyAsync(hC, dC, 20 * sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
        

            // Fine 모델 추론 (coarse 모델 추론과 동시에 진행)
            fineC[preC]->setBindingDimensions(idxFi[0], Dims4{1, 3, 32, 32});  // 예시로 fine 모델이 0번이라 가정
            cudaMemcpyAsync(dIn, images[i].data(), S * sizeof(float), cudaMemcpyHostToDevice, streams[1]);
            
            void* bufF[] = {dIn, dF};
            fineC[preC]->enqueueV2(bufF, streams[1], nullptr);
            cudaMemcpyAsync(hF, dF, 5 * sizeof(float), cudaMemcpyDeviceToHost, streams[1]);
        
            // coarse 모델 결과가 끝난 후 처리
            cudaStreamSynchronize(streams[0]);  // coarse 모델 추론 완료를 기다림
            float* sc = hC;
            int pC = std::distance(sc, std::max_element(sc, sc + 20));
        
            // coarse가 preC인 경우, fine 모델을 미리 추론
            if (pC == preC) {
                hit++;
                cudaStreamSynchronize(streams[1]);  // fine 모델 추론 완료를 기다림
                std::copy(hF, hF + 5, fineResults[pC].begin());
        
                // fine 모델의 결과를 사용
                int loc = std::distance(hF, std::max_element(hF, hF + 5));
                int pF = coarse2fine[pC][loc];
        
                if (pC == coarseLabels[i]) corrCo++;
                if (pF == fineLabels[i]) corrFi++;
            } else {
                // coarse가 0이 아닌 경우, fine 모델 추론을 별도로 실행
                cudaStreamSynchronize(streams[1]);  // fine 모델 추론 완료를 기다림
                
                fineC[pC]->setBindingDimensions(idxFi[pC], Dims4{1,3,32,32});
                cudaMemcpy(dIn, images[i].data(), S*sizeof(float), cudaMemcpyHostToDevice);

                void* bufF[]={dIn,dF};
                fineC[pC]->enqueueV2(bufF,0,nullptr);
                cudaMemcpy(hF, dF, 5*sizeof(float), cudaMemcpyDeviceToHost);
                int loc = std::distance(hF, std::max_element(hF, hF + 5));
                int pF = coarse2fine[pC][loc];

                preC = pC;
        
                if (pC == coarseLabels[i]) corrCo++;
                if (pF == fineLabels[i]) corrFi++;
            }
        
            // 다음 배치 위해 원래 배치 크기로 복원
            coarseC->setBindingDimensions(idxCo, dCo);
        
            // 스트림 해제
            cudaStreamDestroy(streams[0]);
            cudaStreamDestroy(streams[1]);
        }
        
        

        auto t1=std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double,std::milli>(t1-t0).count();
        std::cout<<"Coarse Accuracy: "<<(100.0*corrCo/images.size())<<"%\n";
        std::cout<<"Fine Accuracy:   "<<(100.0*corrFi/images.size())<<"%\n";
        std::cout<<"hit ratio:       "<<(100.0*hit/images.size())<<"%\n";
        std::cout<<"Time total: "<<dt<<" ms, avg "<<dt/images.size()<<" ms/img\n";

        // 정리
        for(auto c : fineC) c->destroy();
        for(auto e : fineE) e->destroy();
        coarseC->destroy(); coarseE->destroy();
        cudaFree(dIn); cudaFree(dC); cudaFree(dF);
        cudaFreeHost(hIn); cudaFreeHost(hC); cudaFreeHost(hF);
    }
    catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}