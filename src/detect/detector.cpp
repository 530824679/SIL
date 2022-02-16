#include "detector.h"

namespace perception
{
    Detector::Detector(const std::string& model_path, const bool& is_gpu, const cv::Size& input_size)
    {
        preProcessor_  = std::make_shared<PreProcess>();
        postProcessor_  = std::make_shared<PostProcess>();
        decode_  = std::make_shared<Decode>();

        loadONNX(model_path, is_gpu, input_size);
    }

    void Detector::loadONNX(const std::string model_path, const bool is_gpu, const cv::Size input_size)
    {
        // 加载设备
        std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
        auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
        OrtCUDAProviderOptions cudaOption;

        sessionOptions = Ort::SessionOptions();
        if (is_gpu && (cudaAvailable == availableProviders.end()))
        {
            std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
            std::cout << "Inference device: CPU" << std::endl;
        }
        else if (is_gpu && (cudaAvailable != availableProviders.end()))
        {
            std::cout << "Inference device: GPU" << std::endl;
            sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
        }
        else
        {
            std::cout << "Inference device: CPU" << std::endl;
        }

        // 加载模型
        env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "SIL_ODET");
        session = Ort::Session(env, model_path.c_str(), sessionOptions);

        // 检查图像宽高是否动态设置
        Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
        std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        preProcessor_->isDynamicInputShape = false;

        if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1)
        {
            std::cout << "Dynamic input shape" << std::endl;
            preProcessor_->isDynamicInputShape = true;
        }

        for (auto shape : inputTensorShape)
            std::cout << "Input shape: " << shape << std::endl;

        // 定义输入输出层
        Ort::AllocatorWithDefaultOptions allocator;

        inputNames.push_back(session.GetInputName(0, allocator));
        outputNames.push_back(session.GetOutputName(0, allocator));
        //outputNames.push_back(session.GetOutputName(1, allocator));
        //outputNames.push_back(session.GetOutputName(2, allocator));
        std::cout << "Input name: " << inputNames[0] << std::endl;
        std::cout << "Output name: " << outputNames[0] << std::endl;
        preProcessor_->inputImageShape = cv::Size2f(input_size);
    }

    std::vector<BoxInfo> Detector::detect(cv::Mat& image, const float& confThreshold, const float& iouThreshold)
    {
        // 为加载的图片准备一个输入的tensor
        float *blob = nullptr;
        std::vector<int64_t> inputTensorShape {1, 3, -1, -1};

        // 预处理输入图片
        preProcessor_->preprocessing2D(image, blob, inputTensorShape);

        size_t inputTensorSize = preProcessor_->vectorProduct(inputTensorShape);
        std::cout << "inputTensorSize: " << inputTensorSize << std::endl;
        std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

        // 从数据中创建输入张量
        std::vector<Ort::Value> inputTensors;
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputTensorShape.data(), inputTensorShape.size()));

        // 模型推理得到输出张量
        std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr},
                                                                  inputNames.data(),
                                                                  inputTensors.data(),
                                                                  1,
                                                                  outputNames.data(),
                                                                  1);

        // 后处理提取出目标坐标、长宽
        cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
        //std::vector<BoxInfo> result = postProcessor_->postprocessing2D(resizedShape, image.size(), outputTensors, confThreshold, iouThreshold);

        int i=0;
        int s[1] = {20};
        for(auto &output :outputTensors)
        {
            auto* output_blob = output.GetTensorData<float>();
            bool ret = decode_->parse_head(output_blob, 0.1, s[i]);
            ++i;
        }

        std::vector<int> indices;
        postProcessor_->nmsBoxes(decode_->origin_rect_, decode_->origin_rect_conf_, 0.1, 0.5, indices);
        std::cout << "amount of NMS indices: " << indices.size() << std::endl;

        std::vector<BoxInfo> detections;

        for (int idx : indices)
        {
            BoxInfo det;
            det.bbox = cv::Rect(decode_->origin_rect_[idx]);
            postProcessor_->scaleCoords(resizedShape, det.bbox, image.size());

            det.score = decode_->origin_rect_conf_[idx];
            det.classId = 0;
            detections.emplace_back(det);
        }




        delete[] blob;

        return detections;
    }


}

