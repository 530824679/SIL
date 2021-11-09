/******************************************************************************/
/*!
File name: runable.h

Description:
This file define function runable

Version: 0.1
Create date: 2021.11.08
Author: Chen Wei
Email: wei.chen@imotion.ai

Copyright (c) iMotion Automotive Technology (Suzhou) Co. Ltd. All rights reserved,
also regarding any disposal, exploitation, reproduction, editing, distribution,
as well as in the event of applications for industrial property rights.
*/
/******************************************************************************/

#include <iostream>
#include "types.h"
#include "cmdline.h"
#include "detector.h"
#include "recogniser.h"
#include "preprocessor.h"
#include "visualization.h"

using namespace cv;
using namespace std;
using namespace perception;

std::vector<std::string> loadNames(const std::string& path)
{
    // load class names
    std::vector<std::string> classNames;
    std::ifstream infile(path);
    if (infile.good())
    {
        std::string line;
        while (getline (infile, line))
        {
            if (line.back() == '\r')
                line.pop_back();
            classNames.emplace_back(line);
        }
        infile.close();
    }
    else
    {
        std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
    }

    return classNames;
}

int main(int argc, char* argv[])
{
    const float confThreshold = 0.4f;
    const float iouThreshold = 0.4f;

    cmdline::parser cmd;
    cmd.add<std::string>("odet_model_path", 'o', "Path to onnx model.", false, "odet.onnx");
    cmd.add<std::string>("kps_model_path", 'k', "Path to onnx model.", false, "kps.onnx");
    cmd.add<std::string>("image_path", 'i', "Image source to be detected.", false, "bus.jpg");
    cmd.add<std::string>("class_names", 'c', "Path to class names file.", false, "coco.names");
    cmd.add<std::string>("visualize", 'v', "Visual Type.", false, "2D");
    cmd.add("gpu", '\0', "Inference on cuda device.");

    cmd.parse_check(argc, argv);

    bool is_gpu = cmd.exist("gpu");
    const std::string classNamesPath = "/home/chenwei/HDD/Project/SIL/model/coco.names";//cmd.get<std::string>("class_names");
    const std::vector<std::string> classNames = loadNames(classNamesPath);
    const std::string imagePath = "/home/chenwei/HDD/Project/SIL/test/bus.jpg";//cmd.get<std::string>("image_path");
    const std::string modelPath = "/home/chenwei/HDD/Project/SIL/model/yolov5m.onnx";//cmd.get<std::string>("model_path");
    const std::string visualType = "2D";

    if (classNames.empty())
    {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }

    // Object Detection Step
    cv::Mat image;
    std::vector<BoxInfo> result;

    try
    {
        Detector detector(modelPath, is_gpu, cv::Size(640, 640));
        std::cout << "Model was initialized." << std::endl;

        // 加载推理图片
        image = cv::imread(imagePath);
        result = detector.detect(image, confThreshold, iouThreshold);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }


    vector<int> input_size(256, 128);
    std::vector<KeypointsInfo> KPS;

    for (int i = 0; i < result.size();i++)
    {
        try
        {
            Recogniser recogniser(modelPath, is_gpu, cv::Size(640, 640), input_size);
            std::cout << "Model was initialized." << std::endl;

            // 加载推理图片
            KPS = recogniser.recognise(image, result[i], confThreshold, iouThreshold);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            return -1;
        }
    }

    // 图像可视化
    Visualizer visualizer;
    static const string winName = "SIL Perception";
    namedWindow(winName, WINDOW_NORMAL);

    if(visualType == "2D")
    {
        visualizer.visualize2D(image, result, classNames);
        cv::imshow(winName, image);
        cv::waitKey(0);
    } else if(visualType == "3D")
    {
        visualizer.visualize3D(image, KPS);
        cv::imshow(winName, image);
        cv::waitKey(0);
    }else{
        //////
    }

    destroyAllWindows();

    return 0;
}