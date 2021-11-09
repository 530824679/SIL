/******************************************************************************/
/*!
File name: visualization.h

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

#ifndef SIL_VISUALIZATION_H
#define SIL_VISUALIZATION_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "types.h"

namespace perception {

    class Visualizer {
    public:
        Visualizer() = default;

        ~Visualizer() {};

        void visualize2D(cv::Mat &image, std::vector<BoxInfo> &detections, const std::vector<std::string> &classNames);

        void visualize3D(cv::Mat &image, std::vector<KeypointsInfo> &detections);

    };
}
#endif //SIL_VISUALIZATION_H
