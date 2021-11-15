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
    cmd.add<std::string>("odet_model_path", 'o', "Path to onnx model.", false, "../model/Keras.onnx");
    cmd.add<std::string>("kps_model_path", 'k', "Path to onnx model.", false, "../model/kps.onnx");
    cmd.add<std::string>("image_path", 'i', "Image source to be detected.", false, "../test/test.mp4");
    cmd.add<std::string>("class_names", 'c', "Path to class names file.", false, "../model/coco.names");
    cmd.add<std::string>("visualize", 'v', "Visual Type.", false, "2D");
    cmd.add("gpu", '\0', "Inference on cuda device.");

    cmd.parse_check(argc, argv);

    bool is_gpu = cmd.exist("gpu");
    const std::string classNamesPath = "../model/coco.names";//cmd.get<std::string>("class_names");
    const std::vector<std::string> classNames = loadNames(classNamesPath);
    const std::string imagePath = "../test/test.mp4";//cmd.get<std::string>("image_path");
    const std::string modelPath = "../model/Keras.onnx";//cmd.get<std::string>("model_path");
    const std::string modelPath1 = "../model/kps.onnx";//cmd.get<std::string>("model_path");
    const std::string visualType = "2D";

    if (classNames.empty())
    {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }

    // Object Detection Step
    cv::Mat image;
    std::vector<BoxInfo> result;
    Visualizer visualizer;

    try
    {
        
        Detector detector(modelPath, is_gpu, cv::Size(640, 640));
        std::cout << "Model was initialized." << std::endl;
         

        // 加载推理图片
        cv::Mat frame;
        std::cout <<  "input image or video filename" << std::endl;

        const std::string  file_ext = imagePath.substr(imagePath.find_last_of(".") + 1);
        //如果输入是图像
        if (file_ext == "png")
	    {
            cv::Mat frame = imread(imagePath);
            PreProcess PreProcess;
            result = detector.detect(frame, confThreshold, iouThreshold);
        }
        //如果输入是视频
        else
        {
            cv::VideoCapture capture(imagePath);
            if (!capture.isOpened())
            {
                std::cout << "open video error"<< std::endl;;
            }
                /*CV_CAP_PROP_POS_MSEC – 视频的当前位置（毫秒）
                CV_CAP_PROP_POS_FRAMES – 视频的当前位置（帧）
                CV_CAP_PROP_FRAME_WIDTH – 视频流的宽度
                CV_CAP_PROP_FRAME_HEIGHT – 视频流的高度
                CV_CAP_PROP_FPS – 帧速率（帧 / 秒）*/
            int frame_width = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
            int frame_height = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
            float frame_fps = capture.get(CV_CAP_PROP_FPS);
            int frame_number = capture.get(CV_CAP_PROP_FRAME_COUNT);//总帧数
            std::cout << "frame_width is " << frame_width<< std::endl;
            std::cout << "frame_height is " << frame_height << std::endl;
            std::cout << "frame_fps is " << frame_fps << std::endl;
            std::cout << "frame_number is " << frame_number << std::endl;

            while (true)
            {
                capture.read(frame);//从视频中读取一个帧
                PreProcess PreProcess;
                result = detector.detect(frame, confThreshold, iouThreshold);
                visualizer.visualize2D(frame, result, classNames);
                cv::imshow("test", frame);
                cv::waitKey(2);

                bool bSuccess = capture.read(frame);
                
                if (!bSuccess)
                {
                    std::cout << "Cannot read frames from video file" << std::endl;
                    break;
                }
            }
        
        }
        // image = cv::imread(imagePath);
        // PreProcess PreProcess;
        // image = PreProcess.data_input(imagePath);
        // cv::imshow("test",image);
        // cv::waitKey(0);
        // result = detector.detect(image, confThreshold, iouThreshold);
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
    //         Recogniser recogniser(modelPath1, is_gpu, cv::Size(640, 640), input_size);
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