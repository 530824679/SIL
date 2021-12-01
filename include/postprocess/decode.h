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
        Decode() = default;
        ~Decode(){};

        double sigmoid(double x);

        std::vector<int> get_anchors(int grid_size);
    };
}


#endif //SIL_POSTPROCESSOR_H
