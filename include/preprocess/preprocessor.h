/******************************************************************************/
/*!
File name: preprocessor.h

Description:
This file define .

Version: 0.1
Create date: 2021.11.08
Author: Chen Wei
Email: wei.chen@imotion.ai

Copyright (c) iMotion Automotive Technology (Suzhou) Co. Ltd. All rights reserved,
also regarding any disposal, exploitation, reproduction, editing, distribution,
as well as in the event of applications for industrial property rights.
*/
/******************************************************************************/

#ifndef SIL_PREPROCESSOR_H
#define SIL_PREPROCESSOR_H

#include "types.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace perception {

    class PreProcess {
    public:
        PreProcess() = default;
        ~PreProcess(){};

        size_t vectorProduct(const std::vector<int64_t>& vector);

        void letterbox(const cv::Mat& image, cv::Mat& outImage,
                       const cv::Size& newShape,
                       const cv::Scalar& color,
                       bool auto_,
                       bool scaleFill,
                       bool scaleUp,
                       int stride);

        void preprocessing2D(cv::Mat &image, float*& blob, std::vector<int64_t>& inputTensorShape);

        void preprocessing3D(cv::Mat &image, BoxInfo box, float* output, std::vector<int> inputTensorShape);

        cv::Mat data_input(const std::string& filename);

    public:
        bool isDynamicInputShape{};
        cv::Size2f inputImageShape;
    };
}

#endif //SIL_PREPROCESSOR_H
