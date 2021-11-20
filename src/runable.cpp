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

#include "../include/runable.hpp"

using namespace cv;
using namespace std;
using namespace perception;

namespace perception
{
    odet_sil::odet_sil()
    {
        confThreshold = 0.6f;
        iouThreshold = 0.6f;
    }


    odet_sil::~odet_sil()
    {
        if(this->pdetector != NULL)
        {
            delete this->pdetector;
        }
    }

    std::vector<std::string> odet_sil::loadNames(const std::string& path)
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

    void odet_sil::init()
    {
        cmdline::parser cmd;
    cmd.add<std::string>("odet_model_path", 'o', "Path to onnx model.", false, "../model/yolov5m.onnx");
    cmd.add<std::string>("kps_model_path", 'k', "Path to onnx model.", false, "../model/kps.onnx");
    cmd.add<std::string>("data_path", 'i', "Data source to be detected.", false, "/home/qzx/code/SIL/test/test.mp4");
    cmd.add<std::string>("class_names", 'c', "Path to class names file.", false, "../model/coco.names");
    cmd.add<std::string>("visual_type", 'v', "Visual Type.", false, "2D");
    cmd.add("cpu", '\0', "Inference on cuda device.");

        // cmd.parse_check(argc, argv);

        is_gpu = cmd.exist("cpu");
        classNamesPath = cmd.get<std::string>("class_names");
        classNames = loadNames(classNamesPath);
        //dataPath = cmd.get<std::string>("data_path");
        odetModelPath = cmd.get<std::string>("odet_model_path");
        kpsModelPath = cmd.get<std::string>("kps_model_path");
        visualType = cmd.get<std::string>("visual_type");

        if (classNames.empty())
        {
            assert("Error: Empty class names file.");
        }

        this->pdetector = new Detector(odetModelPath, is_gpu, cv::Size(640, 640));

        std::cout << "Model was initialized." << std::endl;

    } 

    int odet_sil::process(std::string dataPath)
    {
        result.clear();

        cv::Mat frame;
 
        try
        {
            std::cout << "######## 2D detection test ########"<< std::endl;
            // 加载推理图片
            std::cout <<  "input image or video filename" << std::endl;
            const std::string  file_ext = dataPath.substr(dataPath.find_last_of(".") + 1);

            //如果输入是图像
            if (file_ext == "png" || file_ext == "jpg" || file_ext == "jpeg" || file_ext == "bmp")
            {
                cv::Mat frame = imread(dataPath);
                
                result = pdetector->detect(frame, confThreshold, iouThreshold);
                
                // visualizer.visualize2D(frame, result, classNames);
                // cv::imshow("2D visualize test", frame);
                // cv::waitKey(0);

                vector<int> input_size;
                input_size.push_back(256);
                input_size.push_back(192);
                
                std::vector<KeypointsInfo> KPS;

                std::cout << "######## 3D detection test ########"<< std::endl;
                for (int i = 0; i < result.size();i++)
                {
                    try
                    {
                        Recogniser recogniser(kpsModelPath, is_gpu, cv::Size(640, 640), input_size);
                        std::cout << "Model was initialized." << std::endl;

                        // 加载推理图片
                        std::cout << "######## 加载推理图片 ########"<< std::endl; 
                        KPS = recogniser.recognise(frame, result[i], confThreshold, iouThreshold);
                        // cv::Mat frame1 = cv::imread("/home/qzx/code/SIL/test/test_front2.png");
                        visualizer.visualize3D(frame, KPS);
                        cv::imshow("3D visualize test", frame);
                        cv::waitKey(0);
                    }
                    
                    catch(const std::exception& e)
                    {
                        std::cerr << e.what() << std::endl;
                        return -1;
                    }
                }    
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

                    result = (*pdetector).detect(frame, confThreshold, iouThreshold);
                    
                    // visualizer.visualize2D(frame, result, classNames);
                    // cv::imshow("2D visualize test", frame);
                    // cv::waitKey(2);
                    vector<int> input_size;
                    input_size.push_back(256);
                    input_size.push_back(192);
                    
                    std::vector<KeypointsInfo> KPS;
                    std::cout << "######## 3D detection test ########"<< std::endl;
                    for (int i = 0; i < result.size();i++)
                    {
                        try
                        {
                            Recogniser recogniser(kpsModelPath, is_gpu, cv::Size(640, 640), input_size);
                            std::cout << "Model was initialized." << std::endl;

                            // 加载推理图片
                            std::cout << "######## 加载推理图片 ########"<< std::endl; 
                            KPS = recogniser.recognise(frame, result[i], confThreshold, iouThreshold);

                            visualizer.visualize3D(frame, KPS);
                            cv::imshow("3D visualize test", frame);
                            cv::waitKey(2);
                        }
                        
                        catch(const std::exception& e)
                        {
                            std::cerr << e.what() << std::endl;
                            return -1;
                        }
                    }    
                }
            
            }
            else
            {
                assert("Input file format is error.");
            }
        
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            return -1;
        }


        destroyAllWindows();

        return 0;
    }

}
