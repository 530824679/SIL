/******************************************************************************/
/*!
File name: recogniser.h

Description:
This file define class of .

Version: 0.1
Create date: 2021.11.08
Author: Chen Wei
Email: wei.chen@imotion.ai

Copyright (c) iMotion Automotive Technology (Suzhou) Co. Ltd. All rights reserved,
also regarding any disposal, exploitation, reproduction, editing, distribution,
as well as in the event of applications for industrial property rights.
*/
/******************************************************************************/

#ifndef SIL_RECOGNISER_H
#define SIL_RECOGNISER_H

#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "onnxruntime_cxx_api.h"

#include "types.h"
#include "preprocessor.h"
#include "postprocessor.h"

namespace perception {

    class Recogniser {
    public:
        Recogniser() = default;
        Recogniser(const std::string& model_path, const bool& is_gpu, const cv::Size& input_size);
        ~Recogniser(){};

        std::vector<KeypointsInfo> recognise(cv::Mat& image, const float& confThreshold, const float& iouThreshold);

    private:
        Ort::Env env{nullptr};
        Ort::SessionOptions sessionOptions{nullptr};
        Ort::Session session{nullptr};

        std::vector<const char*> inputNames;
        std::vector<const char*> outputNames;

        std::shared_ptr<PreProcess> preProcessor_;
        std::shared_ptr<PostProcess> postProcessor_;
    };
}


#endif //SIL_RECOGNISER_H
