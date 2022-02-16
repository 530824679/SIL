/******************************************************************************/
/*!
File name: decode.h

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

#ifndef SIL_DECODE_H
#define SIL_DECODE_H

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "onnxruntime_cxx_api.h"
#include "types.h"

namespace perception {

    class Decode {
    public:
        Decode();
        ~Decode();

        bool parse_head(const float* output_blob, float cof_threshold, int feature_size);

        float sigmoid(float x);

        std::vector<int> get_anchors(int grid_size);

    public:
        std::vector<cv::Rect> origin_rect_;                     //保存原始的框信息
        std::vector<float> origin_rect_conf_;               //保存框对应的置信度信息
        int feature_size_[3];

        double _cof_threshold;                         //置信度阈值,框置信度乘以物品种类置信度
        double _nms_area_threshold;                    //nms最小重叠面积阈值
    };
}


#endif //SIL_POSTPROCESSOR_H
