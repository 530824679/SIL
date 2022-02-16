#include "decode.h"
#include <algorithm>

#include <iostream>
#include <iomanip>
#include <fstream>
using namespace std;
using namespace perception;

namespace perception
{
    Decode::Decode()
    {
        int temp[3] = {80, 40, 20};
        memcpy(feature_size_,temp,sizeof(temp));
    }

    Decode::~Decode()
    {

    }

    float Decode::sigmoid(float x)
    {
        return (1 / (1 + exp(-x)));
    }

    std::vector<int> Decode::get_anchors(int grid_size)
    {
        std::vector<int> anchors;
        int anchor_80[6] = {10,13, 16,30, 33,23};
        int anchor_40[6] = {30,61, 62,45, 59,119};
        int anchor_20[6] = {116,90, 156,198, 373,326};

        if(grid_size == 80){
            anchors.insert(anchors.begin(), anchor_80, anchor_80 + 6);
        }
        else if(grid_size == 40){
            anchors.insert(anchors.begin(), anchor_40, anchor_40 + 6);
        }
        else if(grid_size == 20){
            anchors.insert(anchors.begin(), anchor_20, anchor_20 + 6);
        }
        return anchors;
    }

    bool Decode::parse_head(const float* output_blob, float cof_threshold, int feature_size)
    {
//        ofstream outfile;
//        outfile.open("test.txt");
//        if(outfile){
//            int len = feature_size * feature_size * 18;
//            for(int i = 0; i < len; i++)
//            {
//                outfile<<*(output_blob + i)<<endl;
//            }
//
//
//        }
//        outfile.close();




        std::vector<int> anchors = get_anchors(feature_size);

        // item = 4 + 1 + cls
        int item_size = 6;
        size_t anchor_n = 3;
        for(int n = 0; n < anchor_n; ++n)
        {
            for(int i = 0; i < feature_size; ++i)
            {
                for(int j=0; j < feature_size; ++j)
                {
                    float box_prob = output_blob[n * feature_size * feature_size * item_size + feature_size * feature_size * 4 + i * feature_size + j];
                    box_prob = sigmoid(box_prob);

                    // 框置信度不满足整体置信度
                    if(box_prob < cof_threshold)
                        continue;

                    //输出的中心点坐标转化为角点坐标
                    float x = output_blob[n * feature_size * feature_size * item_size + feature_size * feature_size * 0 + i * feature_size + j];
                    float y = output_blob[n * feature_size * feature_size * item_size + feature_size * feature_size * 1 + i * feature_size + j];
                    float w = output_blob[n * feature_size * feature_size * item_size + feature_size * feature_size * 2 + i * feature_size + j];
                    float h = output_blob[n * feature_size * feature_size * item_size + feature_size * feature_size * 3 + i * feature_size + j];

                    float max_prob = 0;
                    int idx = 0;
                    float ttv = 0.0;
                    for(int t = 5;t < item_size; ++t){
                        int tt = n * feature_size * feature_size * item_size + feature_size * feature_size * t + i * feature_size + j;
                        //printf("index is %d\n", tt);
                        float tp= output_blob[n * feature_size * feature_size * item_size + feature_size * feature_size * t + i * feature_size + j];
                        ttv = tp;
                        tp = sigmoid(tp);
                        if(tp > max_prob){
                            max_prob = tp;
                            idx = t;
                            //printf("idx is [%d\t] prob is [%f]\t ttv is %f\n", idx, max_prob, ttv);
                        }
                    }

                    //对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
                    float cof = box_prob * max_prob;
                    if(cof < cof_threshold)
                        continue;

                    x = (sigmoid(x) * 2 - 0.5 + j) * 640.0f / feature_size;
                    y = (sigmoid(y) * 2 - 0.5 + i) * 640.0f / feature_size;
                    w = pow(sigmoid(w) * 2,2) * anchors[n * 2];
                    h = pow(sigmoid(h) * 2,2) * anchors[n * 2 + 1];

                    float r_x = x - w/2;
                    float r_y = y - h/2;
                    cv::Rect rect = cv::Rect(round(r_x),round(r_y),round(w),round(h));
                    origin_rect_.push_back(rect);
                    origin_rect_conf_.push_back(cof);
                }
            }
        }

        if(origin_rect_.size() == 0)
            return false;
        else
            return true;
    }

//    bool Decode::parse_head(const float* output_blob, float cof_threshold, int feature_size)
//    {
//        std::vector<int> anchors = get_anchors(feature_size);
//
//        // item = 4 + 1 + cls
//        int item_size = 85;
//        size_t anchor_n = 3;
//        for(int n = 0; n < anchor_n; ++n)
//        {
//            for(int i = 0; i < feature_size; ++i)
//            {
//                for(int j=0; j < feature_size; ++j)
//                {
//                    float box_prob = output_blob[n * feature_size * feature_size * item_size + i * feature_size * item_size + j * item_size+ 4];
//                    box_prob = sigmoid(box_prob);
//
//                    // 框置信度不满足整体置信度
//                    if(box_prob < cof_threshold)
//                        continue;
//
//                    //输出的中心点坐标转化为角点坐标
//                    float x = output_blob[n * feature_size * feature_size * item_size + i * feature_size * item_size + j * item_size + 0];
//                    float y = output_blob[n * feature_size * feature_size * item_size + i * feature_size * item_size + j * item_size + 1];
//                    float w = output_blob[n * feature_size * feature_size * item_size + i * feature_size * item_size + j * item_size + 2];
//                    float h = output_blob[n * feature_size * feature_size * item_size + i * feature_size * item_size + j * item_size + 3];
//
//                    float max_prob = 0;
//                    int idx = 0;
//                    for(int t = 5;t < item_size; ++t){
//                        float tp= output_blob[n * feature_size * feature_size * item_size + i * feature_size * item_size + j * item_size + t];
//                        tp = sigmoid(tp);
//                        if(tp > max_prob){
//                            max_prob = tp;
//                            idx = t;
//                        }
//                    }
//
//                    //对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
//                    float cof = box_prob * max_prob;
//                    if(cof < cof_threshold)
//                        continue;
//
//                    x = (sigmoid(x) * 2 - 0.5 + j) * 640.0f / feature_size;
//                    y = (sigmoid(y) * 2 - 0.5 + i) * 640.0f / feature_size;
//                    w = pow(sigmoid(w) * 2,2) * anchors[n * 2];
//                    h = pow(sigmoid(h) * 2,2) * anchors[n * 2 + 1];
//
//                    float r_x = x - w/2;
//                    float r_y = y - h/2;
//                    cv::Rect rect = cv::Rect(round(r_x),round(r_y),round(w),round(h));
//                    origin_rect_.push_back(rect);
//                    origin_rect_conf_.push_back(cof);
//                }
//            }
//        }
//
//        if(origin_rect_.size() == 0)
//            return false;
//        else
//            return true;
//    }
}
