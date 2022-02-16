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

    enum Category
    {
        CAR        = 0,
        BUS        = 1,
        PEDESTRIAN = 2,
        UNKNOWN    = 3,
    };

    typedef struct BoxInfo
    {
        cv::Rect bbox;
        float score{};
        int classId;
    } BoxInfo;

    typedef struct KeypointsInfo
    {
        cv::Point2d pt1;
        cv::Point2d pt2;
        cv::Point2d pt3;
        cv::Point2d pt4;
    } KeypointsInfo;

    typedef struct TrackerInfo
    {
        int id = 0;

        std::string label;
        float       score = 0.0;
        bool        valid = false;

        float center_x = 0.0;
        float center_y = 0.0;

        float width  = 0.0;
        float height = 0.0;

        // velocity
        float velocity_x = 0.0;
        float velocity_y = 0.0;

        // acceleration
        float acceleration_x = 0.0;
        float acceleration_y = 0.0;

        float pre_velocity_x = 0.0;
        float pre_velocity_y = 0.0;

        float sum_velocity_x = 0.0;
        float sum_velocity_y = 0.0;

        int year = 0; // the age of tracker
    }TrackerInfo;






//    struct Bbox
//    {
//        float score;
//        int x1;
//        int y1;
//        int w;
//        int h;
//    };
//
//    struct BoundingBox
//    {
//        Rect_<float> rect;
//        BoundingBox(const Bbox &box)
//        {
//            rect = Rect_<float>(box.x1, box.y1, box.w, box.h);
//        }
//    };
//
//    typedef struct TrackingBox
//    {
//        int frame;
//        int id;
//        Rect_<float> box;
//    } TrackingBox;
}
#endif //SIL_TYPES_H



