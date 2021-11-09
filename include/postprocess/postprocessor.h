/******************************************************************************/
/*!
File name: postprocessor.h

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

#ifndef SIL_POSTPROCESSOR_H
#define SIL_POSTPROCESSOR_H

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "onnxruntime_cxx_api.h"
#include "types.h"

namespace perception {

    class PostProcess {
    public:
        PostProcess() = default;
        ~PostProcess(){};

        std::vector<BoxInfo> postprocessing2D(const cv::Size& resizedImageShape, const cv::Size& originalImageShape, std::vector<Ort::Value>& outputTensors, const float& confThreshold, const float& iouThreshold);

        void postprocessing3D(const float *hm, const float *wh, const float *ang, const float *reg,
                         vector<Detection> &result, const int w, const int h, const int classes,
                         const int kernel_size,const float visthresh);

        void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses, float& bestConf, int& bestClassId);

        void scaleCoords(const cv::Size& imageShape, cv::Rect& box, const cv::Size& imageOriginalShape);

        float get_iou_value(cv::Rect rect1, cv::Rect rect2);

        void nmsBoxes(std::vector<cv::Rect>& boxes, std::vector<float>& confs, const float& confThreshold, const float& iouThreshold, std::vector<int>& indices);
    };
}


#endif //SIL_POSTPROCESSOR_H
