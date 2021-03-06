/******************************************************************************/
/*!
File name: detector.h

Description:
This file define class of Detect to get object 2d bounding box.

Version: 0.1
Create date: 2021.11.08
Author: Chen Wei
Email: wei.chen@imotion.ai

Copyright (c) iMotion Automotive Technology (Suzhou) Co. Ltd. All rights reserved,
also regarding any disposal, exploitation, reproduction, editing, distribution,
as well as in the event of applications for industrial property rights.
*/
/******************************************************************************/

#ifndef SIL_DETECTOR_H
#define SIL_DETECTOR_H

#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "onnxruntime_cxx_api.h"

#include "types.h"
#include "preprocessor.h"
#include "postprocessor.h"
#include "decode.h"

namespace perception
{
    class Detector
    {
    public:
        Detector() = default;
        Detector(const std::string& model_path, const bool& is_gpu, const cv::Size& input_size);
        ~Detector(){};

        void loadONNX(const std::string model_path, const bool is_gpu, const cv::Size input_size);
        std::vector<BoxInfo> detect(cv::Mat& image, const float& confThreshold, const float& iouThreshold);

    private:

        Ort::Env env{nullptr};
        Ort::SessionOptions sessionOptions{nullptr};
        Ort::Session session{nullptr};

        std::vector<const char*> inputNames;
        std::vector<const char*> outputNames;

        std::shared_ptr<PreProcess> preProcessor_;
        std::shared_ptr<PostProcess> postProcessor_;
        std::shared_ptr<Decode> decode_;

    };
}
#endif //SIL_DETECTOR_H
