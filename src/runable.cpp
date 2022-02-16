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
        if(pdetector != NULL)
        {
            delete pdetector;
        }

        if(precogniser != NULL)
        {
            delete precogniser;
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
        cmd.add<std::string>("odet_model_path", 'o', "Path to onnx model.", false, "../model/odet-bdd-only-feature_map_20.onnx");
        cmd.add<std::string>("kps_model_path", 'k', "Path to onnx model.", false, "../model/KPS.onnx");
        cmd.add<std::string>("class_names", 'c', "Path to class names file.", false, "../config/coco.names");
        cmd.add<std::string>("visual_type", 'v', "Visual Type.", false, "2D");
        cmd.add("cpu", '\0', "Inference on cuda device.");

        is_gpu = cmd.exist("cpu");
        classNamesPath = cmd.get<std::string>("class_names");
        odetModelPath = cmd.get<std::string>("odet_model_path");
        kpsModelPath = cmd.get<std::string>("kps_model_path");
        visualType = cmd.get<std::string>("visual_type");

        classNames = loadNames(classNamesPath);
        if (classNames.empty())
        {
            assert("Error: Empty class names file.");
        }

        pdetector = new Detector(odetModelPath, is_gpu, cv::Size(640, 640));
        std::cout << "ODET Model was initialized." << std::endl;

        vector<int> input_size;
        input_size.push_back(256);
        input_size.push_back(192);
        precogniser = new Recogniser(kpsModelPath, is_gpu, cv::Size(640, 640), input_size);
        std::cout << "KPS Model was initialized." << std::endl;

        //ptracker = new Track(10);
    }

    int odet_sil::process(std::string dataPath)
    {
        result.clear();

        cv::Mat frame;
        try
        {
            // 加载数据
            std::cout <<  "input image or video filename" << std::endl;
            const std::string file_ext = dataPath.substr(dataPath.find_last_of(".") + 1);

            //如果输入是图像
            if (file_ext == "png" || file_ext == "jpg" || file_ext == "jpeg" || file_ext == "bmp")
            {
                cv::Mat frame = imread(dataPath);
                result = pdetector->detect(frame, confThreshold, iouThreshold);
                
                visualizer.visualize2D(frame, result, classNames);
                cv::imshow("2D visualize test", frame);
                cv::waitKey(0);




//                vector<TrackingBox> detFrameData;
//                for (int i = 0; i < boxes.size(); ++i)
//                {
//                    TrackingBox cur_box;
//                    cur_box.box = boxes[i].rect;
//                    cur_box.id = i;
//                    cur_box.frame = frame_id;
//                    detFrameData.push_back(cur_box);
//                }
//                ++frame_id;
//                vector<TrackingBox> tracking_results = tracker.update(detFrameData);







//                kps.clear();
//                for (int i = 0; i < result.size();i++)
//                {
//                    try
//                    {
//                        // 加载推理图片
//                        kps = precogniser->recognise(frame, result[i], confThreshold, iouThreshold);
//                        //visualizer.visualize3D(frame, kps);
//                        //cv::imshow("3D visualize test", frame);
//                        //cv::waitKey(10);
//                    }
//                    catch(const std::exception& e)
//                    {
//                        std::cerr << e.what() << std::endl;
//                        return -1;
//                    }
//                }
            }


            //如果输入是视频
            else if(file_ext == "avi" || file_ext == "mp4")
            {
                cv::VideoCapture capture(dataPath);
                if (!capture.isOpened())
                {
                    std::cout << "open video error"<< std::endl;;
                }

                while (true)
                {
                    bool bSuccess = capture.read(frame);
                    if (!bSuccess)
                    {
                        std::cout << "Cannot read frames from video file" << std::endl;
                        break;
                    }

                    result.clear();
                    result = pdetector->detect(frame, confThreshold, iouThreshold);

                    for(std::vector<BoxInfo>::iterator it = result.begin(); it != result.end(); ++it)
                    {
                        BoxInfo info = *(it);
                        if (info.classId != 2)
                        {
                            result.erase(it);
                            it--;
                        }
                    }

                    visualizer.visualize2D(frame, result, classNames);
                    cv::imshow("2D visualize test", frame);
                    cv::waitKey(10);

//                    for (int i = 0; i < result.size();i++)
//                    {
//                        try
//                        {
//                            // 加载推理图片
//                            kps = precogniser->recognise(frame, result[i], confThreshold, iouThreshold);
//
////                            visualizer.visualize3D(frame, KPS);
////                            cv::imshow("3D visualize test", frame);
////                            cv::waitKey(2);
//                        }
//                        catch(const std::exception& e)
//                        {
//                            std::cerr << e.what() << std::endl;
//                            return -1;
//                        }
//                    }
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
