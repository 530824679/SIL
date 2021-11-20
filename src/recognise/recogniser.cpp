#include "recogniser.h"
#include "visualization.h"
#include <map>
namespace perception
{
    Recogniser::Recogniser(const std::string& model_path, const bool& is_gpu, const cv::Size& image_size, std::vector<int>& input_size)
    {
        preProcessor_  = std::make_shared<PreProcess>();
        postProcessor_  = std::make_shared<PostProcess>();

        inputShape = input_size;
        loadONNX(model_path, is_gpu, image_size);
    }

    void Recogniser::loadONNX(const std::string model_path, const bool is_gpu, const cv::Size image_size)
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
        size_t input_tensor_size = 1;
        for (int i = 0; i < session.GetInputCount(); i++) {
            // print input node names
            char* input_name = session.GetInputName(i, allocator);
            printf("Input %d : name=%s\n", i, input_name);

            inputNames.push_back(input_name);

            // print input node types
            Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            ONNXTensorElementDataType type = tensor_info.GetElementType();
            printf("Input %d : type=%d\n", i, type);

            // print input shapes/dims
            input_node_dims = tensor_info.GetShape();
            printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
            for (int j = 0; j < input_node_dims.size(); j++) {
                printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
                input_tensor_size *= input_node_dims[j];
            }
        }

        for (int i = 0; i < session.GetOutputCount(); i++) 
        {
            char* output_name = session.GetOutputName(i, allocator);
            printf("Output %d : name=%s\n", i, output_name);

            outputNames.push_back(output_name);

            Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            ONNXTensorElementDataType type = tensor_info.GetElementType();
            printf("Output %d : type=%d\n", i, type);

            output_node_dims = tensor_info.GetShape();
            output_dim_[output_name] = output_node_dims;
            printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
            for (int j = 0; j < output_node_dims.size(); j++) {
                printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
            }
        }


        std::cout << "Input name: " << inputNames[0] << std::endl;
        std::cout << "Output name: " << outputNames[0] << std::endl;
        

        preProcessor_->inputImageShape = cv::Size2f(image_size);

    }

    std::vector<KeypointsInfo> Recogniser::recognise(cv::Mat &image, BoxInfo box, const float &confThreshold, const float &iouThreshold)
    {

        // 为加载的图片准备一个输入的tensor
        float *blob = new float[1047456];
        std::vector<int64_t> inputTensorShape {1, 3, 192, 256};

        // 预处理输入图片
        
        preProcessor_->preprocessing3D(image, box, blob, inputShape);
        
        size_t inputTensorSize = preProcessor_->vectorProduct(inputTensorShape);
        
        std::cout << "inputTensorSize: " << inputTensorSize << std::endl;
        std::vector<float> inputTensorValues(blob, blob + inputTensorSize);
        
        std::vector<Ort::Value> inputTensors;

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
                memoryInfo, inputTensorValues.data(), inputTensorSize,
                inputTensorShape.data(), inputTensorShape.size()
        ));

        // 模型推理
        std::cout << "######## LShape模型推理 ########"<< std::endl;
        std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr},
                                                                  inputNames.data(),
                                                                  inputTensors.data(),
                                                                  1,
                                                                  outputNames.data(),
                                                                  1);

        std::cout << "######## LShape后处理 ########"<< std::endl;        
        // 后处理提取出目标坐标、长宽
        cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
        std::vector<KeypointsInfo> result = postProcessor_->postprocessing3D(resizedShape, image.size(), outputTensors, output_dim_[outputNames[0]],box);
        delete[] blob;

        return result;
    }
}