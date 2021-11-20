/******************************************************************************/
/*!
File name: odet_sil.h

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

#ifndef SIL_ODET_SIL_H
#define SIL_ODET_SIL_H

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include "types.h"
#include "cmdline.h"
#include "detector.h"
#include "recogniser.h"
#include "preprocessor.h"
#include "visualization.h"

namespace perception
{
    class odet_sil {
    public:
        odet_sil();
        ~odet_sil();

        void init();
        int process(std::string dataPath);
        std::vector<std::string> loadNames(const std::string& path);
        
    private:
        bool is_gpu;
        std::string classNamesPath;
        std::vector<std::string> classNames;
        // std::string dataPath;
        std::string odetModelPath;
        std::string kpsModelPath;
        std::string visualType;
        float confThreshold;
        float iouThreshold;

        std::vector<BoxInfo> result;
        Detector *pdetector = NULL;
        PreProcess preProcess;
        Visualizer visualizer;

    };
};

#endif //SIL_RECOGNISER_H
