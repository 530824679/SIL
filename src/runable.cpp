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
    cmd.add<std::string>("odet_model_path", 'o', "Path to onnx model.", false, "../model/yolov5m.onnx");
    cmd.add<std::string>("kps_model_path", 'k', "Path to onnx model.", false, "../model/kps.onnx");
    cmd.add<std::string>("data_path", 'i', "Data source to be detected.", false, "/home/chenwei/HDD/Project/Tracking/KCF/data/video/sample_2.avi");
    cmd.add<std::string>("class_names", 'c', "Path to class names file.", false, "../model/coco.names");
    cmd.add<std::string>("visual_type", 'v', "Visual Type.", false, "2D");
    cmd.add("cpu", '\0', "Inference on cuda device.");

    cmd.parse_check(argc, argv);

    bool is_gpu = cmd.exist("cpu");
    const std::string classNamesPath = cmd.get<std::string>("class_names");
    const std::vector<std::string> classNames = loadNames(classNamesPath);
    const std::string dataPath = cmd.get<std::string>("data_path");
    const std::string odetModelPath = cmd.get<std::string>("odet_model_path");
    const std::string kpsModelPath = cmd.get<std::string>("kps_model_path");
    const std::string visualType = cmd.get<std::string>("visual_type");

    if (classNames.empty())
    {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }

    // Object Detection Step
    cv::Mat image;
    Visualizer visualizer;
    std::vector<BoxInfo> result;

    try
    {
        Detector detector(odetModelPath, is_gpu, cv::Size(640, 640));
        std::cout << "Model was initialized." << std::endl;

        // 加载推理图片
        cv::Mat frame;
        std::cout <<  "input image or video filename" << std::endl;

        const std::string  file_ext = dataPath.substr(dataPath.find_last_of(".") + 1);

        //如果输入是图像
        if (file_ext == "png" || file_ext == "jpg" || file_ext == "jpeg" || file_ext == "bmp")
	    {
            cv::Mat frame = imread(dataPath);
            PreProcess PreProcess;
            result = detector.detect(frame, confThreshold, iouThreshold);
        }
        //如果输入是视频
        else if(file_ext == "avi" || file_ext == "mp4")
        {
            cv::VideoCapture capture(dataPath);
            if (!capture.isOpened())
            {
                std::cout << "open video error"<< std::endl;;
            }

            int frame_width = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
            int frame_height = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);
            float frame_fps = capture.get(cv::CAP_PROP_FPS);
            int frame_number = capture.get(cv::CAP_PROP_FRAME_COUNT);
            std::cout << "frame_width is " << frame_width<< std::endl;
            std::cout << "frame_height is " << frame_height << std::endl;
            std::cout << "frame_fps is " << frame_fps << std::endl;
            std::cout << "frame_number is " << frame_number << std::endl;

            while (true)
            {
                bool bSuccess = capture.read(frame);
                if (!bSuccess)
                {
                    std::cout << "Cannot read frames from video file" << std::endl;
                    break;
                }

                PreProcess PreProcess;
                result = detector.detect(frame, confThreshold, iouThreshold);

                visualizer.visualize2D(frame, result, classNames);
                cv::imshow("test", frame);
                cv::waitKey(2);
            }
        
        }
        else
        {
            assert("Input file format is error.");
        }
//        image = cv::imread(imagePath);
//        PreProcess PreProcess;
//        image = PreProcess.data_input(imagePath);
//        cv::imshow("test",image);
//        cv::waitKey(0);
//        result = detector.detect(image, confThreshold, iouThreshold);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    // std::cout << "222222222222222222"; 

    // vector<int> input_size;
    // input_size.push_back(256);
    // input_size.push_back(192);
    
    // std::vector<KeypointsInfo> KPS;

    // for (int i = 0; i < result.size();i++)
    // {
    //     try
    //     {
    //         Recogniser recogniser(kpsModelPath, is_gpu, cv::Size(640, 640), input_size);
    //         std::cout << "Model was initialized." << std::endl;

    //         // 加载推理图片
    //         KPS = recogniser.recognise(image, result[i], confThreshold, iouThreshold);
    //     }
        
    //     catch(const std::exception& e)
    //     {
    //         std::cerr << e.what() << std::endl;
    //         return -1;
    //     }
    // }

    // 图像可视化
    // Visualizer visualizer;
    // static const string winName = "SIL Perception";
    // namedWindow(winName, WINDOW_NORMAL);

    // if(visualType == "2D")
    // {
    //     visualizer.visualize2D(image, result, classNames);
    //     cv::imshow(winName, image);
    //     cv::waitKey(0);
    // }
    // else if(visualType == "3D")
    // {
    //     visualizer.visualize3D(image, KPS);
    //     cv::imshow(winName, image);
    //     cv::waitKey(0);
    // }
    // else{
    //     //////
    // }

    destroyAllWindows();

    return 0;
}