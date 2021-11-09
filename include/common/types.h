/******************************************************************************/
/*!
File name: types.h

Description:
This file define common types.

Version: 0.1
Create date: 2021.11.08
Author: Chen Wei
Email: wei.chen@imotion.ai

Copyright (c) iMotion Automotive Technology (Suzhou) Co. Ltd. All rights reserved,
also regarding any disposal, exploitation, reproduction, editing, distribution,
as well as in the event of applications for industrial property rights.
*/
/******************************************************************************/

#ifndef SIL_TYPES_H
#define SIL_TYPES_H

#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace perception
{
    #define PI 3.14159265358979323846

    typedef struct BoxInfo
    {
        cv::Rect bbox;
        float score{};
        int classId{};
    } BoxInfo;
}
#endif //SIL_TYPES_H
